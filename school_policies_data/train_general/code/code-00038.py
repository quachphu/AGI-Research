            self._restart_cancellation_in_parent()

            # We only swallow the exception iff it was an AnyIO CancelledError, either
            # directly as exc_val or inside an exception group and there are no cancelled
            # parent cancel scopes visible to us here
            if self._cancel_called and not self._parent_cancellation_is_visible_to_us:
                # For each level-cancel() call made on the host task, call uncancel()
                while self._pending_uncancellations:
                    self._host_task.uncancel()
                    self._pending_uncancellations -= 1

                # Update cancelled_caught and check for exceptions we must not swallow
                if isinstance(exc_val, BaseExceptionGroup):
                    cancelleds_caught, remaining = exc_val.split(
                        lambda exc: (
                            isinstance(exc, CancelledError)
                            and is_anyio_cancellation(exc)
                        )
                    )

                    if cancelleds_caught is None:
                        return False

                    self._cancelled_caught = True

                    if remaining is None:
                        return True

                    context = remaining.__context__
                    try:
                        # Preserve __cause__ and __suppress_context__ by avoiding `raise
                        # ... from ...`
                        raise remaining
                    finally:
                        # Preserve __context__
                        remaining.__context__ = context
                        del context
                else:
                    if isinstance(exc_val, CancelledError) and is_anyio_cancellation(
                        exc_val
                    ):
                        self._cancelled_caught = True
                        return True
                    else:
                        return False
            else:
                if self._pending_uncancellations:
                    assert self._parent_scope is not None
                    assert self._parent_scope._pending_uncancellations is not None
                    self._parent_scope._pending_uncancellations += (
                        self._pending_uncancellations
                    )
                    self._pending_uncancellations = 0

                return False
        finally:
            self._host_task = None
            del exc_val

    @property
    def _effectively_cancelled(self) -> bool:
        cancel_scope: CancelScope | None = self
        while cancel_scope is not None:
            if cancel_scope._cancel_called:
                return True

            if cancel_scope.shield:
                return False

            cancel_scope = cancel_scope._parent_scope

        return False

    @property
    def _parent_cancellation_is_visible_to_us(self) -> bool:
        return (
            self._parent_scope is not None
            and not self.shield
            and self._parent_scope._effectively_cancelled
        )

    def _timeout(self) -> None:
        if self._deadline != math.inf:
            loop = get_running_loop()
            if loop.time() >= self._deadline:
                self.cancel("deadline exceeded")
            else:
                self._timeout_handle = loop.call_at(self._deadline, self._timeout)

    def _deliver_cancellation(self, origin: CancelScope) -> bool:
        """
        Deliver cancellation to directly contained tasks and nested cancel scopes.

        Schedule another run at the end if we still have tasks eligible for
        cancellation.

        :param origin: the cancel scope that originated the cancellation
        :return: ``True`` if the delivery needs to be retried on the next cycle

        """
        should_retry = False
        current = current_task()
        for task in self._tasks:
            # Always skip tasks that are already done (see issue #1111)
            if task.done():
                continue

            should_retry = True
            if task._must_cancel:  # type: ignore[attr-defined]
                continue

            # The task is eligible for cancellation if it has started
            if task is not current and (task is self._host_task or _task_started(task)):
                waiter = task._fut_waiter  # type: ignore[attr-defined]
                if not isinstance(waiter, asyncio.Future) or not waiter.done():
                    task.cancel(origin._cancel_reason)
                    if (
                        task is origin._host_task
                        and origin._pending_uncancellations is not None
                    ):
                        origin._pending_uncancellations += 1

        # Deliver cancellation to child scopes that aren't shielded or running their own
        # cancellation callbacks
        for scope in self._child_scopes:
            if not scope._shield and not scope.cancel_called:
                should_retry = scope._deliver_cancellation(origin) or should_retry

        # Schedule another callback if there are still tasks left
        if origin is self:
            if should_retry:
                self._cancel_handle = get_running_loop().call_soon(
                    self._deliver_cancellation, origin
                )
            else:
                self._cancel_handle = None

        return should_retry

    def _restart_cancellation_in_parent(self) -> None:
        """
        Restart the cancellation effort in the closest directly cancelled parent scope.

        """
        scope = self._parent_scope
        while scope is not None:
            if scope._cancel_called:
                if scope._cancel_handle is None:
                    scope._deliver_cancellation(scope)

                break

            # No point in looking beyond any shielded scope
            if scope._shield:
                break

            scope = scope._parent_scope

    def _reparent(self, new_parent: CancelScope) -> None:
        """
        Move this active scope from its current parent to ``new_parent``.

        Used by :meth:`TaskGroup.start` to move a task that has just called
        ``task_status.started()`` into the target task group's cancel scope.
        """
        if self._parent_scope is new_parent:
            return

        if self._parent_scope is not None:
            self._parent_scope._child_scopes.discard(self)

        self._parent_scope = new_parent
        new_parent._child_scopes.add(self)

        # If the new parent (or an ancestor) is already cancelled, (re)start the
        # delivery loop to ensure we're cancelled at next checkpoint like Trio.
        self._restart_cancellation_in_parent()

    def cancel(self, reason: str | None = None) -> None:
        if not self._cancel_called:
            if self._timeout_handle:
                self._timeout_handle.cancel()
                self._timeout_handle = None

            self._cancel_called = True
            self._cancel_reason = f"Cancelled via cancel scope {id(self):x}"
            if task := current_task():
                self._cancel_reason += f" by {task}"

            if reason:
                self._cancel_reason += f"; reason: {reason}"

            if self._host_task is not None:
                self._deliver_cancellation(self)

    @property
    def deadline(self) -> float:
        return self._deadline

    @deadline.setter
    def deadline(self, value: float) -> None:
        self._deadline = float(value)
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        if self._active and not self._cancel_called:
            self._timeout()

    @property
    def cancel_called(self) -> bool:
        return self._cancel_called

    @property
    def cancelled_caught(self) -> bool:
        return self._cancelled_caught

    @property
    def shield(self) -> bool:
        return self._shield

    @shield.setter
    def shield(self, value: bool) -> None:
        if self._shield != value:
            self._shield = value
            if not value:
                self._restart_cancellation_in_parent()


#
# Task states
#


class TaskState:
    """
    Encapsulates auxiliary task information that cannot be added to the Task instance
    itself because there are no guarantees about its implementation.
    """

    __slots__ = "__weakref__", "cancel_scope", "parent_id"

    def __init__(self, parent_id: int | None, cancel_scope: CancelScope | None):
        self.parent_id = parent_id
        self.cancel_scope = cancel_scope


_task_states: WeakKeyDictionary[asyncio.Task, TaskState] = WeakKeyDictionary()


#
# Task groups
#


class _AsyncioTaskStatus(abc.TaskStatus[T_contra]):
    def __init__(
        self,
        future: asyncio.Future,
        parent_id: int,
        target_scope: CancelScope,
        spawn_scope: CancelScope,
    ):
        self._future = future
        self._parent_id = parent_id
        # The eventual parent scope for this spawn_scope
        # (after task_status.started() has been called)
        self._target_scope = target_scope
        # The task's own cancel scope, also held by its TaskHandle
        self._spawn_scope = spawn_scope

    def started(self, value: T_contra | None = None) -> None:
        task = cast(asyncio.Task, current_task())
        _task_states[task].parent_id = self._parent_id
        if self._future.done():
            if not self._future.cancelled():
                raise RuntimeError("called 'started' twice on the same task status")
            else:
                # Caller of start() was cancelled, nothing to reparent
                return

        self._future.set_result(value)
        self._spawn_scope._reparent(self._target_scope)


if sys.version_info >= (3, 12):
    _eager_task_factory_code: CodeType | None = asyncio.eager_task_factory.__code__
else:
    _eager_task_factory_code = None


class TaskGroup(abc.TaskGroup):
    def __init__(self) -> None:
        self.cancel_scope: CancelScope = CancelScope()
        self._entered = False
        self._exceptions: list[BaseException] = []
        self._tasks: set[asyncio.Task] = set()
        self._on_completed_fut: asyncio.Future[None] | None = None

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("TaskGroup cannot be entered more than once")

        self._entered = True

        self.cancel_scope.__enter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            if exc_val is not None:
                self.cancel_scope.cancel()
                if not isinstance(exc_val, CancelledError):
                    self._exceptions.append(exc_val)

            loop = get_running_loop()
            try:
                if self._tasks:
                    with CancelScope() as wait_scope:
                        while self._tasks:
                            self._on_completed_fut = loop.create_future()

                            try:
                                await self._on_completed_fut
                            except CancelledError as exc:
                                # Shield the scope against further cancellation attempts,
                                # as they're not productive (#695)
                                wait_scope.shield = True
                                self.cancel_scope.cancel()

                                # Set exc_val from the cancellation exception if it was
                                # previously unset. However, we should not replace a native
                                # cancellation exception with one raise by a cancel scope.
                                if exc_val is None or (
                                    isinstance(exc_val, CancelledError)
                                    and not is_anyio_cancellation(exc)
                                ):
                                    exc_val = exc

                            self._on_completed_fut = None
                else:
                    # If there are no child tasks to wait on, run at least one checkpoint
                    # anyway
                    await AsyncIOBackend.cancel_shielded_checkpoint()

                if self._exceptions:
                    # The exception that got us here should already have been
                    # added to self._exceptions so it's ok to break exception
                    # chaining and avoid adding a "During handling of above..."
                    # for each nesting level.
                    raise BaseExceptionGroup(
                        "unhandled errors in a TaskGroup", self._exceptions
                    ) from None
                elif exc_val:
                    raise exc_val
            except BaseException as exc:
                if self.cancel_scope.__exit__(type(exc), exc, exc.__traceback__):
                    return True

                raise

            return self.cancel_scope.__exit__(exc_type, exc_val, exc_tb)
        finally:
            del exc_val, exc_tb, self._exceptions

    def _spawn(
        self,
        coro: Coroutine[Any, Any, T_co],
        name: object,
        task_status: _AsyncioTaskStatus | None = None,
    ) -> TaskHandle[T_co]:
        task_status_future = task_status._future if task_status is not None else None

        def task_done(_task: asyncio.Task) -> None:
            if sys.version_info >= (3, 14) and self.cancel_scope._host_task is not None:
                asyncio.future_discard_from_awaited_by(
                    _task, self.cancel_scope._host_task
                )

            task_state = _task_states[_task]
            assert task_state.cancel_scope is not None
            assert _task in task_state.cancel_scope._tasks
            task_state.cancel_scope._tasks.remove(_task)
            self._tasks.remove(task)
            del _task_states[_task]

            if self._on_completed_fut is not None and not self._tasks:
                try:
                    self._on_completed_fut.set_result(None)
                except asyncio.InvalidStateError:
                    pass

            try:
                exc = _task.exception()
            except CancelledError as e:
                while isinstance(e.__context__, CancelledError):
                    e = e.__context__

                exc = e

            if exc is not None:
                # The future can only be in the cancelled state if the host task was
                # cancelled, so return immediately instead of adding one more
                # CancelledError to the exceptions list
                if task_status_future is not None and task_status_future.cancelled():
                    return

                if task_status_future is None or task_status_future.done():
                    if not isinstance(exc, CancelledError):
                        self._exceptions.append(exc)

                    if not self.cancel_scope._effectively_cancelled:
                        self.cancel_scope.cancel()
                else:
                    task_status_future.set_exception(exc)
            elif task_status_future is not None and not task_status_future.done():
                task_status_future.set_exception(
                    RuntimeError("Child exited without calling task_status.started()")
                )

        if task_status_future is not None:
            parent_id = id(current_task())
            caller_state = _task_states.get(cast(asyncio.Task, current_task()))
            if caller_state is not None and caller_state.cancel_scope is not None:
                initial_scope = caller_state.cancel_scope
            else:
                # The caller is an unmanaged task (no task state)
                initial_scope = self.cancel_scope
        else:
            parent_id = id(self.cancel_scope._host_task)
            initial_scope = self.cancel_scope

        spawn_scope = task_status._spawn_scope if task_status is not None else None
        handle = TaskHandle(coro, name, cancel_scope=spawn_scope)
        loop = asyncio.get_running_loop()
        wrapper_coro = handle._run_coro()
        try:
            if (
                (factory := loop.get_task_factory())
                and getattr(factory, "__code__", None) is _eager_task_factory_code
                and (closure := getattr(factory, "__closure__", None))
            ):
                custom_task_constructor = closure[0].cell_contents
                task = custom_task_constructor(
                    wrapper_coro, loop=loop, name=handle.name
                )
            else:
                task = loop.create_task(wrapper_coro, name=handle.name)
        except BaseException:
            with suppress(BaseException):
                wrapper_coro.close()

            with suppress(BaseException):
                coro.close()

            raise

        # Make the spawned task inherit the initial cancel scope
        _task_states[task] = TaskState(parent_id=parent_id, cancel_scope=initial_scope)
        initial_scope._tasks.add(task)
        self._tasks.add(task)
        if sys.version_info >= (3, 14) and self.cancel_scope._host_task is not None:
            asyncio.future_add_to_awaited_by(task, self.cancel_scope._host_task)

        task.add_done_callback(task_done)
        return handle

    def create_task(
        self,
        coro: Coroutine[Any, Any, T_co],
        *,
        name: object = None,
        context: Context | None = None,
    ) -> TaskHandle[T_co]:
        if not isinstance(coro, Coroutine):
            raise TypeError(f"expected a coroutine, got {coro.__class__.__qualname__}")

        if not self._entered or not self.cancel_scope._active:
            coro.close()
            raise RuntimeError(
                "This task group is not active; no new tasks can be started."
            )

        final_name = get_coro_name(coro, name)

        if context is not None:
            return context.run(self._spawn, coro, name=final_name)
        else:
            return self._spawn(coro, name=final_name)

    async def start(
        self,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        *args: Unpack[PosArgsT],
        name: object = None,
        return_handle: Literal[False, True] = False,
    ) -> Any:
        if not self._entered or not self.cancel_scope._active:
            raise RuntimeError(
                "This task group is not active; no new tasks can be started."
            )

        # Until task_status.started() is called, a task spawned via start() belongs to
        # the *caller's* cancel scope, not the target group's, so cancelling the group
        # does not cancel a task that hasn't reported startup yet. The
        # task_status.started() call moves the task's cancel scope to this task group's
        # cancel scope.
        #
        # The caller may be an unmanaged task (no _task_states entry), in which case
        # fall back to the group's own scope.
        future: asyncio.Future = asyncio.Future()
        final_name = get_callable_name(func, name)
        task_status: _AsyncioTaskStatus[Any] = _AsyncioTaskStatus(
            future, id(self.cancel_scope._host_task), self.cancel_scope, CancelScope()
        )
        coro = call_for_coroutine(func, args, task_status=task_status)
        handle = self._spawn(coro, final_name, task_status)

        # If the task raises an exception after sending a start value without a switch
        # point between, the task group is cancelled and this method never proceeds to
        # process the completed future. That's why we have to have a shielded cancel
        # scope here.
        try:
            await future
        except BaseException:
            if handle.status is TaskHandle.Status.PENDING:
                # Cancel the task and wait for it to exit before returning
                handle.cancel()
                with CancelScope(shield=True):
                    await handle.wait()

            raise

        if return_handle:
            handle._start_value = future.result()
            return handle
        else:
            return future.result()


#
# Threads
#

_Retval_Queue_Type = tuple[T_Retval | None, BaseException | None]


class WorkerThread(Thread):
    MAX_IDLE_TIME = 10  # seconds

    def __init__(
        self,
        root_task: asyncio.Task,
        workers: set[WorkerThread],
        idle_workers: deque[WorkerThread],
    ):
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 14):
            kwargs["context"] = Context()

        super().__init__(name="AnyIO worker thread", **kwargs)
        self.root_task = root_task
        self.workers = workers
        self.idle_workers = idle_workers
        self.loop = root_task._loop
        self.queue: Queue[
            tuple[Context, Callable, tuple, asyncio.Future, CancelScope] | None
        ] = Queue(2)
        self.idle_since = AsyncIOBackend.current_time()
        self.stopping = False

    def _report_result(
        self, future: asyncio.Future, result: Any, exc: BaseException | None
    ) -> None:
        self.idle_since = AsyncIOBackend.current_time()
        if not self.stopping:
            self.idle_workers.append(self)

        if not future.cancelled():
            if exc is not None:
                if isinstance(exc, StopIteration):
                    new_exc = RuntimeError("coroutine raised StopIteration")
                    new_exc.__cause__ = exc
                    exc = new_exc

                future.set_exception(exc)
            else:
                future.set_result(result)

    def run(self) -> None:
        with claim_worker_thread(AsyncIOBackend, self.loop):
            while True:
                item = self.queue.get()
                if item is None:
                    # Shutdown command received
                    return

                context, func, args, future, cancel_scope = item
                if not future.cancelled():
                    result = None
                    exception: BaseException | None = None
                    threadlocals.current_cancel_scope = cancel_scope
                    try:
                        result = context.run(func, *args)
                    except BaseException as exc:
                        exception = exc
                    finally:
                        del threadlocals.current_cancel_scope

                    try:
                        self.loop.call_soon_threadsafe(
                            self._report_result, future, result, exception
                        )
                    except RuntimeError:
                        if not self.loop.is_closed():
                            raise

                    del result, exception

                self.queue.task_done()
                del item, context, func, args, future, cancel_scope

    def stop(self, f: asyncio.Task | None = None) -> None:
        self.stopping = True
        self.queue.put_nowait(None)
        self.workers.discard(self)
        try:
            self.idle_workers.remove(self)
        except ValueError:
            pass


_threadpool_idle_workers: RunVar[deque[WorkerThread]] = RunVar(
    "_threadpool_idle_workers"
)
_threadpool_workers: RunVar[set[WorkerThread]] = RunVar("_threadpool_workers")


#
# Subprocesses
#


@dataclass(eq=False)
class StreamReaderWrapper(abc.ByteReceiveStream):
    _stream: asyncio.StreamReader

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        data = await self._stream.read(max_bytes)
        if data:
            return data
        else:
            raise EndOfStream

    async def aclose(self) -> None:
        self._stream.set_exception(ClosedResourceError())
        await AsyncIOBackend.checkpoint()


@dataclass(eq=False)
class StreamWriterWrapper(abc.ByteSendStream):
    _stream: asyncio.StreamWriter
    _closed: bool = field(init=False, default=False)

    async def send(self, item: bytes) -> None:
        await AsyncIOBackend.checkpoint_if_cancelled()
        stream_paused = self._stream._protocol._paused  # type: ignore[attr-defined]
        try:
            self._stream.write(item)
            await self._stream.drain()
        except (ConnectionResetError, BrokenPipeError, RuntimeError) as exc:
            # If closed by us and/or the peer:
            # * on stdlib, drain() raises ConnectionResetError or BrokenPipeError
            # * on uvloop and Winloop, write() eventually starts raising RuntimeError
            if self._closed:
                raise ClosedResourceError from exc
            elif self._stream.is_closing():
                raise BrokenResourceError from exc

            raise

        if not stream_paused:
            await AsyncIOBackend.cancel_shielded_checkpoint()

    async def aclose(self) -> None:
        self._closed = True
        self._stream.close()
        await AsyncIOBackend.checkpoint()


@dataclass(eq=False)
class Process(abc.Process):
    _process: asyncio.subprocess.Process
    _stdin: StreamWriterWrapper | None
    _stdout: StreamReaderWrapper | None
    _stderr: StreamReaderWrapper | None
    _exited: asyncio.Event
    _transport: asyncio.SubprocessTransport

    async def aclose(self) -> None:
        with CancelScope(shield=True) as scope:
            # We need to close the underlying pipe_transports as well to allow a
            # process blocking on full buffers to receive SIGPIPE and exit.
            if self._stdin:
                await self._stdin.aclose()
                if pipe := self._transport.get_pipe_transport(0):
                    pipe.close()
            if self._stdout:
                await self._stdout.aclose()
                if pipe := self._transport.get_pipe_transport(1):
                    pipe.close()
            if self._stderr:
                await self._stderr.aclose()
                if pipe := self._transport.get_pipe_transport(2):
                    pipe.close()

            scope.shield = False
            try:
                await self.wait()
            except BaseException:
                scope.shield = True
                # Closing the transport on asyncio also handles sending kill
                self._transport.close()
                await self.wait()
                raise

    async def wait(self) -> int:
        await self._exited.wait()
        assert self._process.returncode is not None
        return self._process.returncode

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def send_signal(self, signal: int) -> None:
        self._process.send_signal(signal)

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    @property
    def stdin(self) -> abc.ByteSendStream | None:
        return self._stdin

    @property
    def stdout(self) -> abc.ByteReceiveStream | None:
        return self._stdout

    @property
    def stderr(self) -> abc.ByteReceiveStream | None:
        return self._stderr


def _forcibly_shutdown_process_pool_on_exit(
    workers: set[Process], _task: object
) -> None:
    """
    Forcibly shuts down worker processes belonging to this event loop."""
    child_watcher: asyncio.AbstractChildWatcher | None = None  # type: ignore[name-defined]
    if sys.version_info < (3, 12):
        try:
            child_watcher = asyncio.get_event_loop_policy().get_child_watcher()
        except NotImplementedError:
            pass

    # Close as much as possible (w/o async/await) to avoid warnings
    for process in workers.copy():
        if process.returncode is not None:
            continue

        process._stdin._stream._transport.close()  # type: ignore[union-attr]
        process._stdout._stream._transport.close()  # type: ignore[union-attr]
        process._stderr._stream._transport.close()  # type: ignore[union-attr]
        process.kill()
        if child_watcher:
            child_watcher.remove_child_handler(process.pid)


async def _shutdown_process_pool_on_exit(workers: set[abc.Process]) -> None:
    """
    Shuts down worker processes belonging to this event loop.

    NOTE: this only works when the event loop was started using asyncio.run() or
    anyio.run().

    """
    process: abc.Process
    try:
        await sleep(math.inf)
    except asyncio.CancelledError:
        workers = workers.copy()
        for process in workers:
            if process.returncode is None:
                process.kill()

        for process in workers:
            await process.aclose()


#
# Sockets and networking
#


class StreamProtocol(asyncio.Protocol):
    read_queue: deque[bytes]
    read_event: asyncio.Event
    write_event: asyncio.Event
    exception: Exception | None = None
    is_at_eof: bool = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.read_queue = deque()
        self.read_event = asyncio.Event()
        self.write_event = asyncio.Event()
        self.write_event.set()
        cast(asyncio.Transport, transport).set_write_buffer_limits(0)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc:
            self.exception = exc

        self.read_event.set()
        self.write_event.set()

    def data_received(self, data: bytes) -> None:
        # ProactorEventloop sometimes sends bytearray instead of bytes
        self.read_queue.append(bytes(data))
        self.read_event.set()

    def eof_received(self) -> bool | None:
        self.is_at_eof = True
        self.read_event.set()
        return True

    def pause_writing(self) -> None:
        self.write_event = asyncio.Event()

    def resume_writing(self) -> None:
        self.write_event.set()


class DatagramProtocol(asyncio.DatagramProtocol):
    read_queue: deque[tuple[bytes, IPSockAddrType]]
    read_event: asyncio.Event
    write_event: asyncio.Event
    closed_event: asyncio.Event
    exception: Exception | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.read_queue = deque(maxlen=100)  # arbitrary value
        self.read_event = asyncio.Event()
        self.write_event = asyncio.Event()
        self.closed_event = asyncio.Event()
        self.write_event.set()

    def connection_lost(self, exc: Exception | None) -> None:
        self.read_event.set()
        self.write_event.set()
        self.closed_event.set()

    def datagram_received(self, data: bytes, addr: IPSockAddrType) -> None:
        addr = convert_ipv6_sockaddr(addr)
        self.read_queue.append((data, addr))
        self.read_event.set()

    def error_received(self, exc: Exception) -> None:
        self.exception = exc

    def pause_writing(self) -> None:
        self.write_event.clear()

    def resume_writing(self) -> None:
        self.write_event.set()


class SocketStream(abc.SocketStream):
    def __init__(self, transport: asyncio.Transport, protocol: StreamProtocol):
        self._transport = transport
        self._protocol = protocol
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")
        self._closed = False

    @property
    def _raw_socket(self) -> socket.socket:
        return self._transport.get_extra_info("socket")

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        with self._receive_guard:
            if (
                not self._protocol.read_event.is_set()
                and not self._transport.is_closing()
                and not self._protocol.is_at_eof
            ):
                self._transport.resume_reading()
                await self._protocol.read_event.wait()
                self._transport.pause_reading()
            else:
                await AsyncIOBackend.checkpoint()

            try:
                chunk = self._protocol.read_queue.popleft()
            except IndexError:
                if self._closed:
                    raise ClosedResourceError from None
                elif self._protocol.exception:
                    raise BrokenResourceError from self._protocol.exception
                else:
                    raise EndOfStream from None

            if len(chunk) > max_bytes:
                # Split the oversized chunk
                chunk, leftover = chunk[:max_bytes], chunk[max_bytes:]
                self._protocol.read_queue.appendleft(leftover)

            # If the read queue is empty, clear the flag so that the next call will
            # block until data is available
            if not self._protocol.read_queue:
                self._protocol.read_event.clear()

        return chunk

    async def send(self, item: bytes) -> None:
        with self._send_guard:
            await AsyncIOBackend.checkpoint()

            if self._closed:
                raise ClosedResourceError
            elif self._protocol.exception is not None:
                raise BrokenResourceError from self._protocol.exception

            try:
                self._transport.write(item)
            except RuntimeError as exc:
                if self._transport.is_closing():
                    raise BrokenResourceError from exc
                else:
                    raise

            await self._protocol.write_event.wait()

    async def send_eof(self) -> None:
        try:
            self._transport.write_eof()
        except OSError:
            pass

    async def aclose(self) -> None:
        self._closed = True
        if not self._transport.is_closing():
            try:
                self._transport.write_eof()
            except OSError:
                pass

            self._transport.close()
            await sleep(0)
            self._transport.abort()


class _RawSocketMixin:
    _receive_future: asyncio.Future | None = None
    _send_future: asyncio.Future | None = None
    _closing = False

    def __init__(self, raw_socket: socket.socket):
        self.__raw_socket = raw_socket
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    @property
    def _raw_socket(self) -> socket.socket:
        return self.__raw_socket

    def _wait_until_readable(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        def callback(f: object) -> None:
            del self._receive_future
            loop.remove_reader(self.__raw_socket)

        f = self._receive_future = asyncio.Future()
        loop.add_reader(self.__raw_socket, f.set_result, None)
        f.add_done_callback(callback)
        return f

    def _wait_until_writable(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        def callback(f: object) -> None:
            del self._send_future
            loop.remove_writer(self.__raw_socket)

        f = self._send_future = asyncio.Future()
        loop.add_writer(self.__raw_socket, f.set_result, None)
        f.add_done_callback(callback)
        return f

    async def aclose(self) -> None:
        if not self._closing:
            self._closing = True
            if self.__raw_socket.fileno() != -1:
                self.__raw_socket.close()

            if self._receive_future and not self._receive_future.done():
                self._receive_future.set_result(None)
            if self._send_future and not self._send_future.done():
                self._send_future.set_result(None)


class UNIXSocketStream(_RawSocketMixin, abc.UNIXSocketStream):
    async def send_eof(self) -> None:
        with self._send_guard:
            self._raw_socket.shutdown(socket.SHUT_WR)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._receive_guard:
            while True:
                try:
                    data = self._raw_socket.recv(max_bytes)
                except BlockingIOError:
                    await self._wait_until_readable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    if not data:
                        raise EndOfStream

                    return data

    async def send(self, item: bytes) -> None:
        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._send_guard:
            view = memoryview(item)
            while view:
                try:
                    bytes_sent = self._raw_socket.send(view)
                except BlockingIOError:
                    await self._wait_until_writable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    view = view[bytes_sent:]

    async def receive_fds(self, msglen: int, maxfds: int) -> tuple[bytes, list[int]]:
        if not isinstance(msglen, int) or msglen < 0:
            raise ValueError("msglen must be a non-negative integer")
        if not isinstance(maxfds, int) or maxfds < 1:
            raise ValueError("maxfds must be a positive integer")

        loop = get_running_loop()
        fds = array.array("i")
        await AsyncIOBackend.checkpoint()
        with self._receive_guard:
            while True:
                try:
                    message, ancdata, _flags, _addr = self._raw_socket.recvmsg(
                        msglen, socket.CMSG_LEN(maxfds * fds.itemsize)
                    )
                except BlockingIOError:
                    await self._wait_until_readable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    if not message and not ancdata:
                        raise EndOfStream

                    break

        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level != socket.SOL_SOCKET or cmsg_type != socket.SCM_RIGHTS:
                raise RuntimeError(
                    f"Received unexpected ancillary data; message = {message!r}, "
                    f"cmsg_level = {cmsg_level}, cmsg_type = {cmsg_type}"
                )

            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])

        return message, list(fds)

    async def send_fds(self, message: bytes, fds: Collection[int | IOBase]) -> None:
        if not message:
            raise ValueError("message must not be empty")
        if not fds:
            raise ValueError("fds must not be empty")

        loop = get_running_loop()
        filenos: list[int] = []
        for fd in fds:
            if isinstance(fd, int):
                filenos.append(fd)
            elif isinstance(fd, IOBase):
                filenos.append(fd.fileno())

        fdarray = array.array("i", filenos)
        await AsyncIOBackend.checkpoint()
        with self._send_guard:
            while True:
                try:
                    # The ignore can be removed after mypy picks up
                    # https://github.com/python/typeshed/pull/5545
                    self._raw_socket.sendmsg(
                        [message], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fdarray)]
                    )
                    break
                except BlockingIOError:
                    await self._wait_until_writable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc


class TCPSocketListener(abc.SocketListener):
    _accept_scope: CancelScope | None = None
    _closed = False

    def __init__(self, raw_socket: socket.socket):
        self.__raw_socket = raw_socket
        self._loop = cast(asyncio.BaseEventLoop, get_running_loop())
        self._accept_guard = ResourceGuard("accepting connections from")

    @property
    def _raw_socket(self) -> socket.socket:
        return self.__raw_socket

    async def accept(self) -> abc.SocketStream:
        if self._closed:
            raise ClosedResourceError

        with self._accept_guard:
            await AsyncIOBackend.checkpoint()
            with CancelScope() as self._accept_scope:
                try:
                    client_sock, _addr = await self._loop.sock_accept(self._raw_socket)
                except asyncio.CancelledError:
                    # Workaround for https://bugs.python.org/issue41317
                    try:
                        self._loop.remove_reader(self._raw_socket)
                    except (ValueError, NotImplementedError):
                        pass

                    if self._closed:
                        raise ClosedResourceError from None

                    raise
                finally:
                    self._accept_scope = None

        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        transport, protocol = await self._loop.connect_accepted_socket(
            StreamProtocol, client_sock
        )
        return SocketStream(transport, protocol)

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._accept_scope:
            # Workaround for https://bugs.python.org/issue41317
            try:
                self._loop.remove_reader(self._raw_socket)
            except (ValueError, NotImplementedError):
                pass

            self._accept_scope.cancel()
            await sleep(0)

        self._raw_socket.close()


class UNIXSocketListener(abc.SocketListener):
    def __init__(self, raw_socket: socket.socket):
        self.__raw_socket = raw_socket
        self._loop = get_running_loop()
        self._accept_guard = ResourceGuard("accepting connections from")
        self._closed = False

    async def accept(self) -> abc.SocketStream:
        await AsyncIOBackend.checkpoint()
        with self._accept_guard:
            while True:
                try:
                    client_sock, _ = self.__raw_socket.accept()
                    client_sock.setblocking(False)
                    return UNIXSocketStream(client_sock)
                except BlockingIOError:
                    f: asyncio.Future = asyncio.Future()
                    self._loop.add_reader(self.__raw_socket, f.set_result, None)
                    f.add_done_callback(
                        lambda _: self._loop.remove_reader(self.__raw_socket)
                    )
                    await f
                except OSError as exc:
                    if self._closed:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc

    async def aclose(self) -> None:
        self._closed = True
        self.__raw_socket.close()

    @property
    def _raw_socket(self) -> socket.socket:
        return self.__raw_socket


class UDPSocket(abc.UDPSocket):
    def __init__(
        self, transport: asyncio.DatagramTransport, protocol: DatagramProtocol
    ):
        self._transport = transport
        self._protocol = protocol
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")
        self._closed = False

    @property
    def _raw_socket(self) -> socket.socket:
        return self._transport.get_extra_info("socket")

    async def aclose(self) -> None:
        self._closed = True
        if not self._transport.is_closing():
            self._transport.close()

        await self._protocol.closed_event.wait()

    async def receive(self) -> tuple[bytes, IPSockAddrType]:
        with self._receive_guard:
            await AsyncIOBackend.checkpoint()

            # If the buffer is empty, ask for more data
            if not self._protocol.read_queue and not self._transport.is_closing():
                self._protocol.read_event.clear()
                await self._protocol.read_event.wait()

            try:
                return self._protocol.read_queue.popleft()
            except IndexError:
                if self._closed:
                    raise ClosedResourceError from None
                else:
                    raise BrokenResourceError from None

    async def send(self, item: UDPPacketType) -> None:
        with self._send_guard:
            await AsyncIOBackend.checkpoint()
            await self._protocol.write_event.wait()
            if self._closed:
                raise ClosedResourceError
            elif self._transport.is_closing():
                raise BrokenResourceError
            else:
                self._transport.sendto(*item)


class ConnectedUDPSocket(abc.ConnectedUDPSocket):
    def __init__(
        self, transport: asyncio.DatagramTransport, protocol: DatagramProtocol
    ):
        self._transport = transport
        self._protocol = protocol
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")
        self._closed = False

    @property
    def _raw_socket(self) -> socket.socket:
        return self._transport.get_extra_info("socket")

    async def aclose(self) -> None:
        self._closed = True
        if not self._transport.is_closing():
            self._transport.close()

        await self._protocol.closed_event.wait()

    async def receive(self) -> bytes:
        with self._receive_guard:
            await AsyncIOBackend.checkpoint()

            # If the buffer is empty, ask for more data
            if not self._protocol.read_queue and not self._transport.is_closing():
                self._protocol.read_event.clear()
                await self._protocol.read_event.wait()

            try:
                packet = self._protocol.read_queue.popleft()
            except IndexError:
                if self._closed:
                    raise ClosedResourceError from None
                else:
                    raise BrokenResourceError from None

            return packet[0]

    async def send(self, item: bytes) -> None:
        with self._send_guard:
            await AsyncIOBackend.checkpoint()
            await self._protocol.write_event.wait()
            if self._closed:
                raise ClosedResourceError
            elif self._transport.is_closing():
                raise BrokenResourceError
            else:
                self._transport.sendto(item)


class UNIXDatagramSocket(_RawSocketMixin, abc.UNIXDatagramSocket):
    async def receive(self) -> UNIXDatagramPacketType:
        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._receive_guard:
            while True:
                try:
                    data = self._raw_socket.recvfrom(65536)
                except BlockingIOError:
                    await self._wait_until_readable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    return data

    async def send(self, item: UNIXDatagramPacketType) -> None:
        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._send_guard:
            while True:
                try:
                    self._raw_socket.sendto(*item)
                except BlockingIOError:
                    await self._wait_until_writable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    return


class ConnectedUNIXDatagramSocket(_RawSocketMixin, abc.ConnectedUNIXDatagramSocket):
    async def receive(self) -> bytes:
        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._receive_guard:
            while True:
                try:
                    data = self._raw_socket.recv(65536)
                except BlockingIOError:
                    await self._wait_until_readable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    return data

    async def send(self, item: bytes) -> None:
        loop = get_running_loop()
        await AsyncIOBackend.checkpoint()
        with self._send_guard:
            while True:
                try:
                    self._raw_socket.send(item)
                except BlockingIOError:
                    await self._wait_until_writable(loop)
                except OSError as exc:
                    if self._closing:
                        raise ClosedResourceError from None
                    else:
                        raise BrokenResourceError from exc
                else:
                    return


_read_events: RunVar[dict[int, asyncio.Future[bool]]] = RunVar("read_events")
_write_events: RunVar[dict[int, asyncio.Future[bool]]] = RunVar("write_events")


#
# Synchronization
#


class Event(BaseEvent):
    __slots__ = ("_event",)

    def __new__(cls) -> Self:
        return object.__new__(cls)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        if self.is_set():
            await AsyncIOBackend.checkpoint()
        else:
            await self._event.wait()

    def statistics(self) -> EventStatistics:
        return EventStatistics(len(self._event._waiters))


class Lock(BaseLock):
    __slots__ = "_fast_acquire", "_owner_task", "_waiters"

    def __new__(cls, *, fast_acquire: bool = False) -> Self:
        return object.__new__(cls)

    def __init__(self, *, fast_acquire: bool = False) -> None:
        self._fast_acquire = fast_acquire
        self._owner_task: asyncio.Task | None = None
        self._waiters: deque[tuple[asyncio.Task, asyncio.Future]] = deque()

    async def acquire(self) -> None:
        task = cast(asyncio.Task, current_task())
        if self._owner_task is None and not self._waiters:
            await AsyncIOBackend.checkpoint_if_cancelled()
            self._owner_task = task

            # Unless on the "fast path", yield control of the event loop so that other
            # tasks can run too
            if not self._fast_acquire:
                try:
                    await AsyncIOBackend.cancel_shielded_checkpoint()
                except CancelledError:
                    self.release()
                    raise

            return

        if self._owner_task == task:
            raise RuntimeError("Attempted to acquire an already held Lock")

        fut: asyncio.Future[None] = asyncio.Future()
        item = task, fut
        self._waiters.append(item)
        try:
            await fut
        except CancelledError:
            if fut.cancelled():
                try:
                    self._waiters.remove(item)
                except ValueError:
                    pass
            else:
                self.release()

            raise

    def acquire_nowait(self) -> None:
        task = cast(asyncio.Task, current_task())
        if self._owner_task is None and not self._waiters:
            self._owner_task = task
            return

        if self._owner_task is task:
            raise RuntimeError("Attempted to acquire an already held Lock")

        raise WouldBlock

    def locked(self) -> bool:
        return self._owner_task is not None

    def release(self) -> None:
        if self._owner_task != current_task():
            raise RuntimeError("The current task is not holding this lock")

        # A cancelled waiter that already received ownership removes itself from
        # _waiters before calling release(); any cancelled waiter still queued here
        # was cancelled before being woken, so drop it.
        while self._waiters:
            task, fut = self._waiters.popleft()
            if fut.cancelled():
                continue

            self._owner_task = task
            fut.set_result(None)
            return

        self._owner_task = None

    def statistics(self) -> LockStatistics:
        task_info = AsyncIOTaskInfo(self._owner_task) if self._owner_task else None
        return LockStatistics(self.locked(), task_info, len(self._waiters))


class Semaphore(BaseSemaphore):
    __slots__ = "_fast_acquire", "_max_value", "_value", "_waiters"

    def __new__(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> Self:
        return object.__new__(cls)

    def __init__(
        self,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ):
        super().__init__(initial_value, max_value=max_value)
        self._value = initial_value
        self._max_value = max_value
        self._fast_acquire = fast_acquire
        self._waiters: deque[asyncio.Future[None]] = deque()

    async def acquire(self) -> None:
        if self._value > 0 and not self._waiters:
            await AsyncIOBackend.checkpoint_if_cancelled()
            self._value -= 1

            # Unless on the "fast path", yield control of the event loop so that other
            # tasks can run too
            if not self._fast_acquire:
                try:
                    await AsyncIOBackend.cancel_shielded_checkpoint()
                except CancelledError:
                    self.release()
                    raise

            return

        fut: asyncio.Future[None] = asyncio.Future()
        self._waiters.append(fut)
        try:
            await fut
        except CancelledError:
            if fut.cancelled():
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            else:
                self.release()

            raise

    def acquire_nowait(self) -> None:
        if self._value == 0:
            raise WouldBlock

        self._value -= 1

    def release(self) -> None:
        if self._max_value is not None and self._value == self._max_value:
            raise ValueError("semaphore released too many times")

        while self._waiters:
            fut = self._waiters.popleft()
            if fut.cancelled():
                continue

            fut.set_result(None)
            return

        self._value += 1

    @property
    def value(self) -> int:
        return self._value

    @property
    def max_value(self) -> int | None:
        return self._max_value

    def statistics(self) -> SemaphoreStatistics:
        return SemaphoreStatistics(len(self._waiters))


class CapacityLimiter(BaseCapacityLimiter):
    __slots__ = "_borrowers", "_total_tokens", "_wait_queue"

    def __new__(cls, total_tokens: float) -> Self:
        return object.__new__(cls)

    def __init__(self, total_tokens: float):
        self._total_tokens: float = 0
        self._borrowers: set[Any] = set()
        self._wait_queue: OrderedDict[Any, asyncio.Event] = OrderedDict()
        self.total_tokens = total_tokens

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    @property
    def total_tokens(self) -> float:
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        if not isinstance(value, int) and not math.isinf(value):
            raise TypeError("total_tokens must be an int or math.inf")

        if value < 0:
            raise ValueError("total_tokens must be >= 0")

        self._total_tokens = value

        # Notify waiting tasks that they have acquired the limiter while
        # there is spare capacity.
        while self._wait_queue and len(self._borrowers) < self._total_tokens:
            borrower, event = self._wait_queue.popitem(last=False)
            self._borrowers.add(borrower)
            event.set()

    @property
    def borrowed_tokens(self) -> int:
        return len(self._borrowers)

    @property
    def available_tokens(self) -> float:
        return self._total_tokens - len(self._borrowers)

    def _notify_next_waiter(self) -> None:
        """Hand a free token to the next task in line, if any."""
        if self._wait_queue and len(self._borrowers) < self._total_tokens:
            borrower, event = self._wait_queue.popitem(last=False)
            self._borrowers.add(borrower)
            event.set()

    def acquire_nowait(self) -> None:
        self.acquire_on_behalf_of_nowait(current_task())

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        if borrower in self._borrowers:
            raise RuntimeError(
                "this borrower is already holding one of this CapacityLimiter's tokens"
            )

        if self._wait_queue or len(self._borrowers) >= self._total_tokens:
            raise WouldBlock

        self._borrowers.add(borrower)

    async def acquire(self) -> None:
        return await self.acquire_on_behalf_of(current_task())

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        await AsyncIOBackend.checkpoint_if_cancelled()
        try:
            self.acquire_on_behalf_of_nowait(borrower)
        except WouldBlock:
            event = asyncio.Event()
            self._wait_queue[borrower] = event
            try:
                await event.wait()
            except BaseException:
                self._wait_queue.pop(borrower, None)
                if event.is_set():
                    self._borrowers.discard(borrower)
                    self._notify_next_waiter()

                raise
        else:
            try:
                await AsyncIOBackend.cancel_shielded_checkpoint()
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        self.release_on_behalf_of(current_task())

    def release_on_behalf_of(self, borrower: object) -> None:
        try:
            self._borrowers.remove(borrower)
        except KeyError:
            raise RuntimeError(
                "this borrower isn't holding any of this CapacityLimiter's tokens"
            ) from None

        self._notify_next_waiter()

    def statistics(self) -> CapacityLimiterStatistics:
        return CapacityLimiterStatistics(
            self.borrowed_tokens,
            self.total_tokens,
            tuple(self._borrowers),
            len(self._wait_queue),
        )


_default_thread_limiter: RunVar[CapacityLimiter] = RunVar("_default_thread_limiter")


#
# Operating system signals
#


class _SignalReceiver:
    def __init__(self, signals: tuple[Signals, ...]):
        self._signals = signals
        self._loop = get_running_loop()
        self._signal_queue: deque[Signals] = deque()
        self._future: asyncio.Future = asyncio.Future()
        self._handled_signals: set[Signals] = set()

    def _deliver(self, signum: Signals) -> None:
        self._signal_queue.append(signum)
        if not self._future.done():
            self._future.set_result(None)

    def __enter__(self) -> Self:
        for sig in set(self._signals):
            self._loop.add_signal_handler(sig, self._deliver, sig)
            self._handled_signals.add(sig)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for sig in self._handled_signals:
            self._loop.remove_signal_handler(sig)

    def __aiter__(self) -> _SignalReceiver:
        return self

    async def __anext__(self) -> Signals:
        await AsyncIOBackend.checkpoint()
        if not self._signal_queue:
            self._future = asyncio.Future()
            await self._future

        return self._signal_queue.popleft()


#
# Testing and debugging
#


class AsyncIOTaskInfo(TaskInfo):
    def __init__(self, task: asyncio.Task):
        task_state = _task_states.get(task)
        if task_state is None:
            parent_id = None
        else:
            parent_id = task_state.parent_id

        coro = task.get_coro()
        assert coro is not None, "created TaskInfo from a completed Task"
        super().__init__(id(task), parent_id, task.get_name(), coro)
        self._task = weakref.ref(task)

    def has_pending_cancellation(self) -> bool:
        if not (task := self._task()):
            # If the task isn't around anymore, it won't have a pending cancellation
            return False

        if task._must_cancel or (  # type: ignore[attr-defined]
            isinstance(task._fut_waiter, asyncio.Future)  # type: ignore[attr-defined]
            and task._fut_waiter.cancelled()  # type: ignore[attr-defined]
        ):
            return True

        if task_state := _task_states.get(task):
            if cancel_scope := task_state.cancel_scope:
                return cancel_scope._effectively_cancelled

        return False


class TestRunner(abc.TestRunner):
    _send_stream: MemoryObjectSendStream[tuple[Awaitable[Any], asyncio.Future[Any]]]

    def __init__(
        self,
        *,
        debug: bool | None = None,
        use_uvloop: bool = False,
        loop_factory: Callable[[], AbstractEventLoop] | None = None,
    ) -> None:
        if use_uvloop and loop_factory is None:
            if sys.platform != "win32":
                import uvloop

                loop_factory = uvloop.new_event_loop
            else:
                import winloop

                loop_factory = winloop.new_event_loop

        self._runner = Runner(debug=debug, loop_factory=loop_factory)
        self._exceptions: list[BaseException] = []
        self._runner_task: asyncio.Task | None = None

    def __enter__(self) -> Self:
        self._runner.__enter__()
        self.get_loop().set_exception_handler(self._exception_handler)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._runner.__exit__(exc_type, exc_val, exc_tb)

    def get_loop(self) -> AbstractEventLoop:
        return self._runner.get_loop()

    def is_running(self) -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _exception_handler(
        self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        if isinstance(context.get("exception"), Exception):
            self._exceptions.append(context["exception"])
        else:
            loop.default_exception_handler(context)

    def _raise_async_exceptions(self) -> None:
        # Re-raise any exceptions raised in asynchronous callbacks
        if self._exceptions:
            exceptions, self._exceptions = self._exceptions, []
            if len(exceptions) == 1:
                raise exceptions[0]
            elif exceptions:
                raise BaseExceptionGroup(
                    "Multiple exceptions occurred in asynchronous callbacks", exceptions
                )

    async def _run_tests_and_fixtures(
        self,
        receive_stream: MemoryObjectReceiveStream[
            tuple[Awaitable[T_Retval], asyncio.Future[T_Retval]]
        ],
    ) -> None:
        from _pytest.outcomes import OutcomeException

        with receive_stream, self._send_stream:
            async for coro, future in receive_stream:
                try:
                    retval = await coro
                except CancelledError as exc:
                    if not future.cancelled():
                        future.cancel(*exc.args)

                    raise
                except BaseException as exc:
                    if not future.cancelled():
                        future.set_exception(exc)

                    if not isinstance(exc, (Exception, OutcomeException)):
                        raise
                else:
                    if not future.cancelled():
                        future.set_result(retval)

    async def _call_in_runner_task(
        self,
        func: Callable[P, Awaitable[T_Retval]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T_Retval:
        if not self._runner_task:
            self._send_stream, receive_stream = create_memory_object_stream[
                tuple[Awaitable[Any], asyncio.Future]
            ](1)
            self._runner_task = self.get_loop().create_task(
                self._run_tests_and_fixtures(receive_stream)
            )

        coro = func(*args, **kwargs)
        future: asyncio.Future[T_Retval] = self.get_loop().create_future()
        self._send_stream.send_nowait((coro, future))
        return await future

    def run_asyncgen_fixture(
        self,
        fixture_func: Callable[..., AsyncGenerator[T_Retval, Any]],
        kwargs: dict[str, Any],
    ) -> Iterable[T_Retval]:
        asyncgen = fixture_func(**kwargs)
        fixturevalue: T_Retval = self.get_loop().run_until_complete(
            self._call_in_runner_task(asyncgen.asend, None)
        )
        self._raise_async_exceptions()

        yield fixturevalue

        try:
            self.get_loop().run_until_complete(
                self._call_in_runner_task(asyncgen.asend, None)
            )
        except StopAsyncIteration:
            self._raise_async_exceptions()
        else:
            self.get_loop().run_until_complete(asyncgen.aclose())
            raise RuntimeError("Async generator fixture did not stop")

    def run_fixture(
        self,
        fixture_func: Callable[..., Coroutine[Any, Any, T_Retval]],
        kwargs: dict[str, Any],
    ) -> T_Retval:
        retval = self.get_loop().run_until_complete(
            self._call_in_runner_task(fixture_func, **kwargs)
        )
        self._raise_async_exceptions()
        return retval

    def run_test(
        self, test_func: Callable[..., Coroutine[Any, Any, Any]], kwargs: dict[str, Any]
    ) -> None:
        from _pytest.outcomes import OutcomeException

        try:
            self.get_loop().run_until_complete(
                self._call_in_runner_task(test_func, **kwargs)
            )
        except Exception as exc:
            self._exceptions.append(exc)
        except OutcomeException:
            raise
        except BaseException:
            # A BaseException (e.g. KeyboardInterrupt, SystemExit) interrupted the event loop before
            # the test completed. Cancel _runner_task so it does not resume when the event
            # loop is re-entered during async generator fixture teardown.
            if self._runner_task is not None and not self._runner_task.done():
                self._runner_task.cancel()
                self._send_stream.close()
                try:
                    self.get_loop().run_until_complete(self._runner_task)
                except CancelledError:
                    pass
                finally:
                    self._runner_task = None
            raise
        self._raise_async_exceptions()


class _ProcessStreamProtocol(asyncio.subprocess.SubprocessStreamProtocol):
    """
    A subprocess protocol that allows us to be notified of ``process_exited``

    asyncio's own ``Process.wait()`` only resolves once every pipe transport has
    disconnected so to get same semantics as on trio and uvloop we need this.
    """

    def __init__(self) -> None:
        # Match the standard factory for asyncio.create_process
        super().__init__(limit=2**16, loop=asyncio.get_running_loop())
        self.exited = asyncio.Event()

    def process_exited(self) -> None:
        super().process_exited()
        self.exited.set()


class AsyncIOBackend(AsyncBackend):
    @classmethod
    def run(
        cls,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        options: dict[str, Any],
    ) -> T_Retval:
        @wraps(func)
        async def wrapper() -> T_Retval:
            task = cast(asyncio.Task, current_task())
            task.set_name(get_callable_name(func))
            _task_states[task] = TaskState(None, None)

            try:
                return await func(*args)
            finally:
                del _task_states[task]

        debug = options.get("debug", None)
        loop_factory = options.get("loop_factory", None)
        if loop_factory is None and options.get("use_uvloop", False):
            if sys.platform != "win32":
                import uvloop

                loop_factory = uvloop.new_event_loop
            else:
                import winloop

                loop_factory = winloop.new_event_loop

        with Runner(debug=debug, loop_factory=loop_factory) as runner:
            return runner.run(wrapper())

    @classmethod
    def current_token(cls) -> object:
        return get_running_loop()

    @classmethod
    def current_time(cls) -> float:
        return get_running_loop().time()

    @classmethod
    def cancelled_exception_class(cls) -> type[BaseException]:
        return CancelledError

    @classmethod
    async def checkpoint(cls) -> None:
        await sleep(0)

    @classmethod
    async def checkpoint_if_cancelled(cls) -> None:
        task = current_task()
        if task is None:
            return

        try:
            cancel_scope = _task_states[task].cancel_scope
        except KeyError:
            return

        while cancel_scope:
            if cancel_scope.cancel_called:
                await sleep(0)
            elif cancel_scope.shield:
                break
            else:
                cancel_scope = cancel_scope._parent_scope

    @classmethod
    async def cancel_shielded_checkpoint(cls) -> None:
        with CancelScope(shield=True):
            await sleep(0)

    @classmethod
    async def sleep(cls, delay: float) -> None:
        await sleep(delay)

    @classmethod
    def create_cancel_scope(
        cls, *, deadline: float = math.inf, shield: bool = False
    ) -> CancelScope:
        return CancelScope(deadline=deadline, shield=shield)

    @classmethod
    def current_effective_deadline(cls) -> float:
        if (task := current_task()) is None:
            return math.inf

        try:
            cancel_scope = _task_states[task].cancel_scope
        except KeyError:
            return math.inf

        deadline = math.inf
        while cancel_scope:
            deadline = min(deadline, cancel_scope.deadline)
            if cancel_scope._cancel_called:
                deadline = -math.inf
                break
            elif cancel_scope.shield:
                break
            else:
                cancel_scope = cancel_scope._parent_scope

        return deadline

    @classmethod
    def create_task_group(cls) -> abc.TaskGroup:
        return TaskGroup()

    @classmethod
    def create_event(cls) -> BaseEvent:
        return Event()

    @classmethod
    def create_lock(cls, *, fast_acquire: bool) -> BaseLock:
        return Lock(fast_acquire=fast_acquire)

    @classmethod
    def create_semaphore(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> BaseSemaphore:
        return Semaphore(initial_value, max_value=max_value, fast_acquire=fast_acquire)

    @classmethod
    def create_capacity_limiter(cls, total_tokens: float) -> BaseCapacityLimiter:
        return CapacityLimiter(total_tokens)

    @classmethod
    async def run_sync_in_worker_thread(  # type: ignore[return]
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        abandon_on_cancel: bool = False,
        limiter: BaseCapacityLimiter | None = None,
    ) -> T_Retval:
        await cls.checkpoint()

        # If this is the first run in this event loop thread, set up the necessary
        # variables
        try:
            idle_workers = _threadpool_idle_workers.get()
            workers = _threadpool_workers.get()
        except LookupError:
            idle_workers = deque()
            workers = set()
            _threadpool_idle_workers.set(idle_workers)
            _threadpool_workers.set(workers)

        async with limiter or cls.current_default_thread_limiter():
            with CancelScope(shield=not abandon_on_cancel) as scope:
                future = asyncio.Future[T_Retval]()
                root_task = find_root_task()
                if not idle_workers:
                    worker = WorkerThread(root_task, workers, idle_workers)
                    worker.start()
                    workers.add(worker)
                    root_task.add_done_callback(worker.stop, context=Context())
                else:
                    worker = idle_workers.pop()

                    # Prune any other workers that have been idle for MAX_IDLE_TIME
                    # seconds or longer
                    now = cls.current_time()
                    while idle_workers:
                        if (
                            now - idle_workers[0].idle_since
                            < WorkerThread.MAX_IDLE_TIME
                        ):
                            break

                        expired_worker = idle_workers.popleft()
                        expired_worker.root_task.remove_done_callback(
                            expired_worker.stop
                        )
                        expired_worker.stop()

                context = copy_context()
                context.run(set_current_async_library, None)
                if abandon_on_cancel or scope._parent_scope is None:
                    worker_scope = scope
                else:
                    worker_scope = scope._parent_scope

                worker.queue.put_nowait((context, func, args, future, worker_scope))
                return await future

    @classmethod
    def check_cancelled(cls) -> None:
        scope: CancelScope | None = threadlocals.current_cancel_scope
        while scope is not None:
            if scope.cancel_called:
                raise CancelledError(f"Cancelled via cancel scope {id(scope):x}")

            if scope.shield:
                return

            scope = scope._parent_scope

    @classmethod
    def run_async_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_co:
        async def task_wrapper() -> T_co:
            __tracebackhide__ = True
            if scope is not None:
                task = cast(asyncio.Task, current_task())
                _task_states[task] = TaskState(None, scope)
                scope._tasks.add(task)
            try:
                return await func(*args)
            except CancelledError as exc:
                raise concurrent.futures.CancelledError(str(exc)) from None
            finally:
                if scope is not None:
                    scope._tasks.discard(task)

        loop = cast(
            "AbstractEventLoop", token or threadlocals.current_token.native_token
        )
        if loop.is_closed():
            raise RunFinishedError

        context = copy_context()
        context.run(set_current_async_library, "asyncio")
        scope = getattr(threadlocals, "current_cancel_scope", None)
        f: concurrent.futures.Future[T_co] = context.run(
            asyncio.run_coroutine_threadsafe, task_wrapper(), loop=loop
        )
        return f.result()

    @classmethod
    def run_sync_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_Retval:
        @wraps(func)
        def wrapper() -> None:
            try:
                set_current_async_library("asyncio")
                f.set_result(func(*args))
            except BaseException as exc:
                f.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise

        loop = cast(
            "AbstractEventLoop", token or threadlocals.current_token.native_token
        )
        if loop.is_closed():
            raise RunFinishedError

        f: concurrent.futures.Future[T_Retval] = Future()
        loop.call_soon_threadsafe(wrapper)
        return f.result()

    @classmethod
    async def open_process(
        cls,
        command: StrOrBytesPath | Sequence[StrOrBytesPath],
        *,
        stdin: int | IO[Any] | None,
        stdout: int | IO[Any] | None,
        stderr: int | IO[Any] | None,
        **kwargs: Any,
    ) -> Process:
        await cls.checkpoint()
        if isinstance(command, PathLike):
            command = os.fspath(command)

        # Use loop.subprocess_shell()/subprocess_exec() rather than their
        # asyncio.create_subprocess_*() counterparts to get access to
        # transport/protocol.
        loop = asyncio.get_running_loop()
        if isinstance(command, (str, bytes)):
            transport, protocol = await loop.subprocess_shell(
                _ProcessStreamProtocol,
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                **kwargs,
            )
        else:
            transport, protocol = await loop.subprocess_exec(
                _ProcessStreamProtocol,
                *command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                **kwargs,
            )

        process = asyncio.subprocess.Process(transport, protocol, loop)
        stdin_stream = StreamWriterWrapper(process.stdin) if process.stdin else None
        stdout_stream = StreamReaderWrapper(process.stdout) if process.stdout else None
        stderr_stream = StreamReaderWrapper(process.stderr) if process.stderr else None
        return Process(
            process,
            stdin_stream,
            stdout_stream,
            stderr_stream,
            protocol.exited,
            transport,
        )

    @classmethod
    def setup_process_pool_exit_at_shutdown(cls, workers: set[abc.Process]) -> None:
        create_task(
            _shutdown_process_pool_on_exit(workers),
            name="AnyIO process pool shutdown task",
        )
        find_root_task().add_done_callback(
            partial(_forcibly_shutdown_process_pool_on_exit, workers)  # type:ignore[arg-type]
        )

    @classmethod
    async def connect_tcp(
        cls, host: str, port: int, local_address: IPSockAddrType | None = None
    ) -> abc.SocketStream:
        transport, protocol = cast(
            tuple[asyncio.Transport, StreamProtocol],
            await get_running_loop().create_connection(
                StreamProtocol, host, port, local_addr=local_address
            ),
        )
        transport.pause_reading()
        return SocketStream(transport, protocol)

    @classmethod
    async def connect_unix(cls, path: str | bytes) -> abc.UNIXSocketStream:
        await cls.checkpoint()
        loop = get_running_loop()
        raw_socket = socket.socket(socket.AF_UNIX)
        raw_socket.setblocking(False)
        while True:
            try:
                raw_socket.connect(path)
            except BlockingIOError:
                f: asyncio.Future = asyncio.Future()
                loop.add_writer(raw_socket, f.set_result, None)
                f.add_done_callback(lambda _: loop.remove_writer(raw_socket))
                await f
            except BaseException:
                raw_socket.close()
                raise
            else:
                return UNIXSocketStream(raw_socket)

    @classmethod
    def create_tcp_listener(cls, sock: socket.socket) -> SocketListener:
        return TCPSocketListener(sock)

    @classmethod
    def create_unix_listener(cls, sock: socket.socket) -> SocketListener:
        return UNIXSocketListener(sock)

    @classmethod
    async def create_udp_socket(
        cls,
        family: AddressFamily,
        local_address: IPSockAddrType | None,
        remote_address: IPSockAddrType | None,
        reuse_port: bool,
    ) -> UDPSocket | ConnectedUDPSocket:
        transport, protocol = await get_running_loop().create_datagram_endpoint(
            DatagramProtocol,
            local_addr=local_address,
            remote_addr=remote_address,
            family=family,
            reuse_port=reuse_port,
        )
        if protocol.exception:
            transport.close()
            raise protocol.exception

        if not remote_address:
            return UDPSocket(transport, protocol)
        else:
            return ConnectedUDPSocket(transport, protocol)

    @classmethod
    async def create_unix_datagram_socket(  # type: ignore[override]
        cls, raw_socket: socket.socket, remote_path: str | bytes | None
    ) -> abc.UNIXDatagramSocket | abc.ConnectedUNIXDatagramSocket:
        await cls.checkpoint()
        loop = get_running_loop()

        if remote_path:
            while True:
                try:
                    raw_socket.connect(remote_path)
                except BlockingIOError:
                    f: asyncio.Future = asyncio.Future()
                    loop.add_writer(raw_socket, f.set_result, None)
                    f.add_done_callback(lambda _: loop.remove_writer(raw_socket))
                    await f
                except BaseException:
                    raw_socket.close()
                    raise
                else:
                    return ConnectedUNIXDatagramSocket(raw_socket)
        else:
            return UNIXDatagramSocket(raw_socket)

    @classmethod
    async def getaddrinfo(
        cls,
        host: bytes | str | None,
        port: str | int | None,
        *,
        family: int | AddressFamily = 0,
        type: int | SocketKind = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> Sequence[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        return await get_running_loop().getaddrinfo(
            host, port, family=family, type=type, proto=proto, flags=flags
        )

    @classmethod
    async def getnameinfo(
        cls, sockaddr: IPSockAddrType, flags: int = 0
    ) -> tuple[str, str]:
        return await get_running_loop().getnameinfo(sockaddr, flags)

    @classmethod
    async def wait_readable(cls, obj: FileDescriptorLike) -> None:
        try:
            read_events = _read_events.get()
        except LookupError:
            read_events = {}
            _read_events.set(read_events)

        fd = obj if isinstance(obj, int) else obj.fileno()
        if read_events.get(fd):
            raise BusyResourceError("reading from")

        loop = get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()

        def cb() -> None:
            try:
                del read_events[fd]
            except KeyError:
                pass
            else:
                remove_reader(fd)

            try:
                fut.set_result(True)
            except asyncio.InvalidStateError:
                pass

        try:
            loop.add_reader(fd, cb)
        except NotImplementedError:
            from anyio._core._asyncio_selector_thread import get_selector

            selector = get_selector()
            selector.add_reader(fd, cb)
            remove_reader = selector.remove_reader
        else:
            remove_reader = loop.remove_reader

        read_events[fd] = fut
        try:
            success = await fut
        finally:
            try:
                del read_events[fd]
            except KeyError:
                pass
            else:
                remove_reader(fd)

        if not success:
            raise ClosedResourceError

    @classmethod
    async def wait_writable(cls, obj: FileDescriptorLike) -> None:
        try:
            write_events = _write_events.get()
        except LookupError:
            write_events = {}
            _write_events.set(write_events)

        fd = obj if isinstance(obj, int) else obj.fileno()
        if write_events.get(fd):
            raise BusyResourceError("writing to")

        loop = get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()

        def cb() -> None:
            try:
                del write_events[fd]
            except KeyError:
                pass
            else:
                remove_writer(fd)

            try:
                fut.set_result(True)
            except asyncio.InvalidStateError:
                pass

        try:
            loop.add_writer(fd, cb)
        except NotImplementedError:
            from anyio._core._asyncio_selector_thread import get_selector

            selector = get_selector()
            selector.add_writer(fd, cb)
            remove_writer = selector.remove_writer
        else:
            remove_writer = loop.remove_writer

        write_events[fd] = fut
        try:
            success = await fut
        finally:
            try:
                del write_events[fd]
            except KeyError:
                pass
            else:
                remove_writer(fd)

        if not success:
            raise ClosedResourceError

    @classmethod
    def notify_closing(cls, obj: FileDescriptorLike) -> None:
        fd = obj if isinstance(obj, int) else obj.fileno()
        loop = get_running_loop()

        try:
            write_events = _write_events.get()
        except LookupError:
            pass
        else:
            try:
                fut = write_events.pop(fd)
            except KeyError:
                pass
            else:
                try:
                    fut.set_result(False)
                except asyncio.InvalidStateError:
                    pass

                try:
                    loop.remove_writer(fd)
                except NotImplementedError:
                    from anyio._core._asyncio_selector_thread import get_selector

                    get_selector().remove_writer(fd)

        try:
            read_events = _read_events.get()
        except LookupError:
            pass
        else:
            try:
                fut = read_events.pop(fd)
            except KeyError:
                pass
            else:
                try:
                    fut.set_result(False)
                except asyncio.InvalidStateError:
                    pass

                try:
                    loop.remove_reader(fd)
                except NotImplementedError:
                    from anyio._core._asyncio_selector_thread import get_selector

                    get_selector().remove_reader(fd)

    @classmethod
    async def wrap_listener_socket(cls, sock: socket.socket) -> SocketListener:
        if hasattr(socket, "AF_UNIX") and sock.family == socket.AF_UNIX:
            return UNIXSocketListener(sock)

        return TCPSocketListener(sock)

    @classmethod
    async def wrap_stream_socket(cls, sock: socket.socket) -> SocketStream:
        transport, protocol = await get_running_loop().create_connection(
            StreamProtocol, sock=sock
        )
        return SocketStream(transport, protocol)

    @classmethod
    async def wrap_unix_stream_socket(cls, sock: socket.socket) -> UNIXSocketStream:
        return UNIXSocketStream(sock)

    @classmethod
    async def wrap_udp_socket(cls, sock: socket.socket) -> UDPSocket:
        transport, protocol = await get_running_loop().create_datagram_endpoint(
            DatagramProtocol, sock=sock
        )
        return UDPSocket(transport, protocol)

    @classmethod
    async def wrap_connected_udp_socket(cls, sock: socket.socket) -> ConnectedUDPSocket:
        transport, protocol = await get_running_loop().create_datagram_endpoint(
            DatagramProtocol, sock=sock
        )
        return ConnectedUDPSocket(transport, protocol)

    @classmethod
    async def wrap_unix_datagram_socket(cls, sock: socket.socket) -> UNIXDatagramSocket:
        return UNIXDatagramSocket(sock)

    @classmethod
    async def wrap_connected_unix_datagram_socket(
        cls, sock: socket.socket
    ) -> ConnectedUNIXDatagramSocket:
        return ConnectedUNIXDatagramSocket(sock)

    @classmethod
    def current_default_thread_limiter(cls) -> CapacityLimiter:
        try:
            return _default_thread_limiter.get()
        except LookupError:
            limiter = CapacityLimiter(40)
            _default_thread_limiter.set(limiter)
            return limiter

    @classmethod
    def open_signal_receiver(
        cls, *signals: Signals
    ) -> AbstractContextManager[AsyncIterator[Signals]]:
        return _SignalReceiver(signals)

    @classmethod
    def get_current_task(cls) -> TaskInfo:
        return AsyncIOTaskInfo(current_task())  # type: ignore[arg-type]

    @classmethod
    def get_running_tasks(cls) -> Sequence[TaskInfo]:
        return [AsyncIOTaskInfo(task) for task in all_tasks() if not task.done()]

    @classmethod
    async def wait_all_tasks_blocked(cls) -> None:
        await cls.checkpoint()
        this_task = current_task()
        while True:
            for task in all_tasks():
                if task is this_task:
                    continue

                waiter = task._fut_waiter  # type: ignore[attr-defined]
                if waiter is None or waiter.done():
                    await sleep(0.1)
                    break
            else:
                return

    @classmethod
    def create_test_runner(cls, options: dict[str, Any]) -> TestRunner:
        return TestRunner(**options)


backend_class = AsyncIOBackend
<|endoftext|>from __future__ import annotations

import sys
from abc import ABCMeta, abstractmethod
from types import TracebackType
from typing import TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

T = TypeVar("T")


class AsyncResource(metaclass=ABCMeta):
    """
    Abstract base class for all closeable asynchronous resources.

    Works as an asynchronous context manager which returns the instance itself on enter,
    and calls :meth:`aclose` on exit.
    """

    __slots__ = ()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @abstractmethod
    async def aclose(self) -> None:
        """Close the resource."""
<|endoftext|>from __future__ import annotations

import math
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import AbstractContextManager
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind, socket
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike

    from .._core._synchronization import CapacityLimiter, Event, Lock, Semaphore
    from .._core._tasks import CancelScope
    from .._core._testing import TaskInfo
    from ._sockets import (
        ConnectedUDPSocket,
        ConnectedUNIXDatagramSocket,
        IPSockAddrType,
        SocketListener,
        SocketStream,
        UDPSocket,
        UNIXDatagramSocket,
        UNIXSocketStream,
    )
    from ._subprocesses import Process
    from ._tasks import TaskGroup
    from ._testing import TestRunner

T_Retval = TypeVar("T_Retval")
T_co = TypeVar("T_co", covariant=True)
PosArgsT = TypeVarTuple("PosArgsT")
StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]


class AsyncBackend(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def run(
        cls,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        options: dict[str, Any],
    ) -> T_Retval:
        """
        Run the given coroutine function in an asynchronous event loop.

        The current thread must not be already running an event loop.

        :param func: a coroutine function
        :param args: positional arguments to ``func``
        :param kwargs: positional arguments to ``func``
        :param options: keyword arguments to call the backend ``run()`` implementation
            with
        :return: the return value of the coroutine function
        """

    @classmethod
    @abstractmethod
    def current_token(cls) -> object:
        """
        Return an object that allows other threads to run code inside the event loop.

        :return: a token object, specific to the event loop running in the current
            thread
        """

    @classmethod
    @abstractmethod
    def current_time(cls) -> float:
        """
        Return the current value of the event loop's internal clock.

        :return: the clock value (seconds)
        """

    @classmethod
    @abstractmethod
    def cancelled_exception_class(cls) -> type[BaseException]:
        """Return the exception class that is raised in a task if it's cancelled."""

    @classmethod
    @abstractmethod
    async def checkpoint(cls) -> None:
        """
        Check if the task has been cancelled, and allow rescheduling of other tasks.

        This is effectively the same as running :meth:`checkpoint_if_cancelled` and then
        :meth:`cancel_shielded_checkpoint`.
        """

    @classmethod
    async def checkpoint_if_cancelled(cls) -> None:
        """
        Check if the current task group has been cancelled.

        This will check if the task has been cancelled, but will not allow other tasks
        to be scheduled if not.

        """
        if cls.current_effective_deadline() == -math.inf:
            await cls.checkpoint()

    @classmethod
    async def cancel_shielded_checkpoint(cls) -> None:
        """
        Allow the rescheduling of other tasks.

        This will give other tasks the opportunity to run, but without checking if the
        current task group has been cancelled, unlike with :meth:`checkpoint`.

        """
        with cls.create_cancel_scope(shield=True):
            await cls.sleep(0)

    @classmethod
    @abstractmethod
    async def sleep(cls, delay: float) -> None:
        """
        Pause the current task for the specified duration.

        :param delay: the duration, in seconds
        """

    @classmethod
    @abstractmethod
    def create_cancel_scope(
        cls, *, deadline: float = math.inf, shield: bool = False
    ) -> CancelScope:
        pass

    @classmethod
    @abstractmethod
    def current_effective_deadline(cls) -> float:
        """
        Return the nearest deadline among all the cancel scopes effective for the
        current task.

        :return:
            - a clock value from the event loop's internal clock
            - ``inf`` if there is no deadline in effect
            - ``-inf`` if the current scope has been cancelled
        :rtype: float
        """

    @classmethod
    @abstractmethod
    def create_task_group(cls) -> TaskGroup:
        pass

    @classmethod
    @abstractmethod
    def create_event(cls) -> Event:
        pass

    @classmethod
    @abstractmethod
    def create_lock(cls, *, fast_acquire: bool) -> Lock:
        pass

    @classmethod
    @abstractmethod
    def create_semaphore(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> Semaphore:
        pass

    @classmethod
    @abstractmethod
    def create_capacity_limiter(cls, total_tokens: float) -> CapacityLimiter:
        pass

    @classmethod
    @abstractmethod
    async def run_sync_in_worker_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        abandon_on_cancel: bool = False,
        limiter: CapacityLimiter | None = None,
    ) -> T_Retval:
        pass

    @classmethod
    @abstractmethod
    def check_cancelled(cls) -> None:
        pass

    @classmethod
    @abstractmethod
    def run_async_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_co:
        pass

    @classmethod
    @abstractmethod
    def run_sync_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_Retval:
        pass

    @classmethod
    @abstractmethod
    async def open_process(
        cls,
        command: StrOrBytesPath | Sequence[StrOrBytesPath],
        *,
        stdin: int | IO[Any] | None,
        stdout: int | IO[Any] | None,
        stderr: int | IO[Any] | None,
        cwd: StrOrBytesPath | None = None,
        env: Mapping[str, str] | None = None,
        startupinfo: Any = None,
        creationflags: int = 0,
        start_new_session: bool = False,
        pass_fds: Sequence[int] = (),
        user: str | int | None = None,
        group: str | int | None = None,
        extra_groups: Iterable[str | int] | None = None,
        umask: int = -1,
        **kwargs: Any,
    ) -> Process:
        pass

    @classmethod
    @abstractmethod
    def setup_process_pool_exit_at_shutdown(cls, workers: set[Process]) -> None:
        pass

    @classmethod
    @abstractmethod
    async def connect_tcp(
        cls, host: str, port: int, local_address: IPSockAddrType | None = None
    ) -> SocketStream:
        pass

    @classmethod
    @abstractmethod
    async def connect_unix(cls, path: str | bytes) -> UNIXSocketStream:
        pass

    @classmethod
    @abstractmethod
    def create_tcp_listener(cls, sock: socket) -> SocketListener:
        pass

    @classmethod
    @abstractmethod
    def create_unix_listener(cls, sock: socket) -> SocketListener:
        pass

    @classmethod
    @abstractmethod
    async def create_udp_socket(
        cls,
        family: AddressFamily,
        local_address: IPSockAddrType | None,
        remote_address: IPSockAddrType | None,
        reuse_port: bool,
    ) -> UDPSocket | ConnectedUDPSocket:
        pass

    @classmethod
    @overload
    async def create_unix_datagram_socket(
        cls, raw_socket: socket, remote_path: None
    ) -> UNIXDatagramSocket: ...

    @classmethod
    @overload
    async def create_unix_datagram_socket(
        cls, raw_socket: socket, remote_path: str | bytes
    ) -> ConnectedUNIXDatagramSocket: ...

    @classmethod
    @abstractmethod
    async def create_unix_datagram_socket(
        cls, raw_socket: socket, remote_path: str | bytes | None
    ) -> UNIXDatagramSocket | ConnectedUNIXDatagramSocket:
        pass

    @classmethod
    @abstractmethod
    async def getaddrinfo(
        cls,
        host: bytes | str | None,
        port: str | int | None,
        *,
        family: int | AddressFamily = 0,
        type: int | SocketKind = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> Sequence[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        pass

    @classmethod
    @abstractmethod
    async def getnameinfo(
        cls, sockaddr: IPSockAddrType, flags: int = 0
    ) -> tuple[str, str]:
        pass

    @classmethod
    @abstractmethod
    async def wait_readable(cls, obj: FileDescriptorLike) -> None:
        pass

    @classmethod
    @abstractmethod
    async def wait_writable(cls, obj: FileDescriptorLike) -> None:
        pass

    @classmethod
    @abstractmethod
    def notify_closing(cls, obj: FileDescriptorLike) -> None:
        pass

    @classmethod
    @abstractmethod
    async def wrap_listener_socket(cls, sock: socket) -> SocketListener:
        pass

    @classmethod
    @abstractmethod
    async def wrap_stream_socket(cls, sock: socket) -> SocketStream:
        pass

    @classmethod
    @abstractmethod
    async def wrap_unix_stream_socket(cls, sock: socket) -> UNIXSocketStream:
        pass

    @classmethod
    @abstractmethod
    async def wrap_udp_socket(cls, sock: socket) -> UDPSocket:
        pass

    @classmethod
    @abstractmethod
    async def wrap_connected_udp_socket(cls, sock: socket) -> ConnectedUDPSocket:
        pass

    @classmethod
    @abstractmethod
    async def wrap_unix_datagram_socket(cls, sock: socket) -> UNIXDatagramSocket:
        pass

    @classmethod
    @abstractmethod
    async def wrap_connected_unix_datagram_socket(
        cls, sock: socket
    ) -> ConnectedUNIXDatagramSocket:
        pass

    @classmethod
    @abstractmethod
    def current_default_thread_limiter(cls) -> CapacityLimiter:
        pass

    @classmethod
    @abstractmethod
    def open_signal_receiver(
        cls, *signals: Signals
    ) -> AbstractContextManager[AsyncIterator[Signals]]:
        pass

    @classmethod
    @abstractmethod
    def get_current_task(cls) -> TaskInfo:
        pass

    @classmethod
    @abstractmethod
    def get_running_tasks(cls) -> Sequence[TaskInfo]:
        pass

    @classmethod
    @abstractmethod
    async def wait_all_tasks_blocked(cls) -> None:
        pass

    @classmethod
    @abstractmethod
    def create_test_runner(cls, options: dict[str, Any]) -> TestRunner:
        pass
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazyimport import (
    fix_package_names,
    install_lazy_importer,
    set_deprecated_aliases,
)

if TYPE_CHECKING or not install_lazy_importer():
    from ._eventloop import AsyncBackend as AsyncBackend
    from ._resources import AsyncResource as AsyncResource
    from ._sockets import ConnectedUDPSocket as ConnectedUDPSocket
    from ._sockets import ConnectedUNIXDatagramSocket as ConnectedUNIXDatagramSocket
    from ._sockets import IPAddressType as IPAddressType
    from ._sockets import IPSockAddrType as IPSockAddrType
    from ._sockets import SocketAttribute as SocketAttribute
    from ._sockets import SocketListener as SocketListener
    from ._sockets import SocketStream as SocketStream
    from ._sockets import UDPPacketType as UDPPacketType
    from ._sockets import UDPSocket as UDPSocket
    from ._sockets import UNIXDatagramPacketType as UNIXDatagramPacketType
    from ._sockets import UNIXDatagramSocket as UNIXDatagramSocket
    from ._sockets import UNIXSocketStream as UNIXSocketStream
    from ._streams import AnyByteReceiveStream as AnyByteReceiveStream
    from ._streams import AnyByteSendStream as AnyByteSendStream
    from ._streams import AnyByteStream as AnyByteStream
    from ._streams import AnyByteStreamConnectable as AnyByteStreamConnectable
    from ._streams import (
        AnyUnreliableByteReceiveStream as AnyUnreliableByteReceiveStream,
    )
    from ._streams import AnyUnreliableByteSendStream as AnyUnreliableByteSendStream
    from ._streams import AnyUnreliableByteStream as AnyUnreliableByteStream
    from ._streams import ByteReceiveStream as ByteReceiveStream
    from ._streams import ByteSendStream as ByteSendStream
    from ._streams import ByteStream as ByteStream
    from ._streams import ByteStreamConnectable as ByteStreamConnectable
    from ._streams import Listener as Listener
    from ._streams import ObjectReceiveStream as ObjectReceiveStream
    from ._streams import ObjectSendStream as ObjectSendStream
    from ._streams import ObjectStream as ObjectStream
    from ._streams import ObjectStreamConnectable as ObjectStreamConnectable
    from ._streams import UnreliableObjectReceiveStream as UnreliableObjectReceiveStream
    from ._streams import UnreliableObjectSendStream as UnreliableObjectSendStream
    from ._streams import UnreliableObjectStream as UnreliableObjectStream
    from ._subprocesses import Process as Process
    from ._tasks import TaskGroup as TaskGroup
    from ._tasks import TaskStatus as TaskStatus
    from ._testing import TestRunner as TestRunner

    fix_package_names()
    set_deprecated_aliases(
        {
            "CapacityLimiter": "anyio.CapacityLimiter",
            "Condition": "anyio.Condition",
            "Event": "anyio.Event",
            "Lock": "anyio.Lock",
            "Semaphore": "anyio.Semaphore",
            "CancelScope": "anyio.CancelScope",
            "BlockingPortal": "anyio.from_thread.BlockingPortal",
        }
    )
<|endoftext|>from __future__ import annotations

import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Coroutine
from contextvars import Context
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, final, overload

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

if TYPE_CHECKING:
    from .._core._tasks import CancelScope, TaskHandle

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True, default=None)
PosArgsT = TypeVarTuple("PosArgsT")


def get_coro_name(coro: Coroutine[Any, Any, object], override: object = None) -> str:
    if override is not None:
        return str(override)

    try:
        cr_frame = coro.cr_frame  # type: ignore[attr-defined]
        module = cr_frame.f_globals["__name__"]
    except (AttributeError, KeyError):
        module = None

    qualname = getattr(coro, "__qualname__", None)
    return ".".join([x for x in (module, qualname) if x])


def get_callable_name(func: Callable, override: object = None) -> str:
    if override is not None:
        return str(override)

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    return ".".join([x for x in (module, qualname) if x])


def call_for_coroutine(
    func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
    args: tuple[Unpack[PosArgsT]],
    **kwargs: Any,
) -> Coroutine[Any, Any, T_co]:
    """
    Call the given function with the given positional and keyword arguments.

    :return: the resulting coroutine
    :raises TypeError: if the return value was not a coroutine object

    """
    coro = func(*args, **kwargs)
    if not isinstance(coro, Coroutine):
        prefix = f"{func.__module__}." if hasattr(func, "__module__") else ""
        raise TypeError(
            f"Expected {prefix}{func.__qualname__}() to return a coroutine, but "
            f"the return value ({coro!r}) is not a coroutine object"
        )

    return coro


class TaskStatus(Protocol[T_contra]):
    @overload
    def started(self: TaskStatus[None]) -> None: ...

    @overload
    def started(self, value: T_contra) -> None: ...

    def started(self, value: T_contra | None = None) -> None:
        """
        Signal that the task has started.

        :param value: object passed back to the starter of the task
        """


class TaskGroup(metaclass=ABCMeta):
    """
    Groups several asynchronous tasks together.

    :ivar cancel_scope: the cancel scope inherited by all child tasks
    :vartype cancel_scope: CancelScope

    .. note:: On asyncio, support for eager task factories is considered to be
        **experimental**. In particular, they don't follow the usual semantics of new
        tasks being scheduled on the next iteration of the event loop, and may thus
        cause unexpected behavior in code that wasn't written with such semantics in
        mind.
    """

    cancel_scope: CancelScope

    def cancel(self, reason: str | None = None) -> None:
        """
        Cancel this task group's cancel scope immediately.

        This is a shortcut for calling ``.cancel_scope.cancel()`` on the task group.

        :param reason: a message describing the reason for the cancellation

        .. versionadded:: 4.14.0

        """
        self.cancel_scope.cancel(reason)

    @abstractmethod
    def create_task(
        self,
        coro: Coroutine[Any, Any, T_co],
        *,
        name: object = None,
        context: Context | None = None,
    ) -> TaskHandle[T_co]:
        """
        Create a new task from a coroutine object and schedule it to run.

        :param coro: a coroutine object
        :param name: optional name to give the task
        :param context: optional context to run the task in
        :return: a task handle

        .. versionadded:: 4.14.0
        """

    @final
    def start_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> TaskHandle[T_co]:
        """
        Start a new task in this task group.

        :param func: a coroutine function
        :param args: positional arguments to call the function with
        :param name: name of the task, for the purposes of introspection and debugging
        :return: a task handle

        .. versionadded:: 3.0
        .. versionchanged:: 4.14.0
            This method now returns a task handle.

        """
        final_name = get_callable_name(func, name)
        return self.create_task(call_for_coroutine(func, args), name=final_name)

    @overload
    async def start(
        self,
        func: Callable[..., Coroutine[Any, Any, T_co]],
        *args: object,
        name: object = None,
        return_handle: Literal[False] = ...,
    ) -> Any: ...

    @overload
    async def start(
        self,
        func: Callable[..., Coroutine[Any, Any, T_co]],
        *args: object,
        name: object = None,
        return_handle: Literal[True],
    ) -> TaskHandle[T_co, Any]: ...

    @abstractmethod
    async def start(
        self,
        func: Callable[..., Coroutine[Any, Any, T_co]],
        *args: object,
        name: object = None,
        return_handle: Literal[False, True] = False,
    ) -> Any:
        """
        Start a new task and wait until it signals for readiness.

        The target callable must accept a keyword argument ``task_status`` (of type
        :class:`TaskStatus`). Awaiting on this method will return whatever was passed to
        ``task_status.started()`` (``None`` by default).

        .. note:: The :class:`TaskStatus` class is generic, and the type argument should
            indicate the type of the value that will be passed to
            ``task_status.started()``.

        :param func: a coroutine function that accepts the ``task_status`` keyword
            argument
        :param args: positional arguments to call the function with
        :param name: an optional name for the task, for introspection and debugging
        :param return_handle: if ``True``, return a :class:`TaskHandle` which also
            contains the start value in ``start_value``
        :return: the value passed to ``task_status.started()``
        :raises RuntimeError: if the task finishes without calling
            ``task_status.started()``

        .. seealso:: :ref:`start_initialize`

        .. versionadded:: 3.0
        """

    @abstractmethod
    async def __aenter__(self) -> TaskGroup:
        """Enter the task group context and allow starting new tasks."""

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the task group context waiting for all tasks to finish."""
<|endoftext|>from __future__ import annotations

import errno
import socket
from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
from contextlib import AsyncExitStack
from io import IOBase
from ipaddress import IPv4Address, IPv6Address
from socket import AddressFamily
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from .._core._eventloop import get_async_backend
from .._core._typedattr import (
    TypedAttributeProvider,
    TypedAttributeSet,
    typed_attribute,
)
from ._streams import ByteStream, Listener, UnreliableObjectStream

if TYPE_CHECKING:
    from ._tasks import TaskGroup

IPAddressType: TypeAlias = str | IPv4Address | IPv6Address
IPSockAddrType: TypeAlias = tuple[str, int]
SockAddrType: TypeAlias = IPSockAddrType | str
UDPPacketType: TypeAlias = tuple[bytes, IPSockAddrType]
UNIXDatagramPacketType: TypeAlias = tuple[bytes, str]
T_Retval = TypeVar("T_Retval")


def _validate_socket(
    sock_or_fd: socket.socket | int,
    sock_type: socket.SocketKind,
    addr_family: socket.AddressFamily = socket.AF_UNSPEC,
    *,
    require_connected: bool = False,
    require_bound: bool = False,
) -> socket.socket:
    if isinstance(sock_or_fd, int):
        try:
            sock = socket.socket(fileno=sock_or_fd)
        except OSError as exc:
            if exc.errno == errno.ENOTSOCK:
                raise ValueError(
                    "the file descriptor does not refer to a socket"
                ) from exc
            elif require_connected:
                raise ValueError("the socket must be connected") from exc
            elif require_bound:
                raise ValueError("the socket must be bound to a local address") from exc
            else:
                raise
    elif isinstance(sock_or_fd, socket.socket):
        sock = sock_or_fd
    else:
        raise TypeError(
            f"expected an int or socket, got {type(sock_or_fd).__qualname__} instead"
        )

    try:
        if require_connected:
            try:
                sock.getpeername()
            except OSError as exc:
                raise ValueError("the socket must be connected") from exc

        if require_bound:
            try:
                if sock.family in (socket.AF_INET, socket.AF_INET6):
                    bound_addr = sock.getsockname()[1]
                else:
                    bound_addr = sock.getsockname()
            except OSError:
                bound_addr = None

            if not bound_addr:
                raise ValueError("the socket must be bound to a local address")

        if addr_family != socket.AF_UNSPEC and sock.family != addr_family:
            raise ValueError(
                f"address family mismatch: expected {addr_family.name}, got "
                f"{sock.family.name}"
            )

        if sock.type != sock_type:
            raise ValueError(
                f"socket type mismatch: expected {sock_type.name}, got {sock.type.name}"
            )
    except BaseException:
        # Avoid ResourceWarning from the locally constructed socket object
        if isinstance(sock_or_fd, int):
            sock.detach()

        raise

    sock.setblocking(False)
    return sock


class SocketAttribute(TypedAttributeSet):
    """
    .. attribute:: family
        :type: socket.AddressFamily

        the address family of the underlying socket

    .. attribute:: local_address
        :type: tuple[str, int] | str

        the local address the underlying socket is connected to

    .. attribute:: local_port
        :type: int

        for IP based sockets, the local port the underlying socket is bound to

    .. attribute:: raw_socket
        :type: socket.socket

        the underlying stdlib socket object

    .. attribute:: remote_address
        :type: tuple[str, int] | str

        the remote address the underlying socket is connected to

    .. attribute:: remote_port
        :type: int

        for IP based sockets, the remote port the underlying socket is connected to
    """

    family: AddressFamily = typed_attribute()
    local_address: SockAddrType = typed_attribute()
    local_port: int = typed_attribute()
    raw_socket: socket.socket = typed_attribute()
    remote_address: SockAddrType = typed_attribute()
    remote_port: int = typed_attribute()


class _SocketProvider(TypedAttributeProvider):
    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        from .._core._sockets import convert_ipv6_sockaddr as convert

        attributes: dict[Any, Callable[[], Any]] = {
            SocketAttribute.family: lambda: self._raw_socket.family,
            SocketAttribute.local_address: lambda: convert(
                self._raw_socket.getsockname()
            ),
            SocketAttribute.raw_socket: lambda: self._raw_socket,
        }
        try:
            peername: tuple[str, int] | None = convert(self._raw_socket.getpeername())
        except OSError:
            peername = None

        # Provide the remote address for connected sockets
        if peername is not None:
            attributes[SocketAttribute.remote_address] = lambda: peername

        # Provide local and remote ports for IP based sockets
        if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
            attributes[SocketAttribute.local_port] = lambda: (
                self._raw_socket.getsockname()[1]
            )
            if peername is not None:
                remote_port = peername[1]
                attributes[SocketAttribute.remote_port] = lambda: remote_port

        return attributes

    @property
    @abstractmethod
    def _raw_socket(self) -> socket.socket:
        pass


class SocketStream(ByteStream, _SocketProvider):
    """
    Transports bytes over a socket.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(cls, sock_or_fd: socket.socket | int) -> SocketStream:
        """
        Wrap an existing socket object or file descriptor as a socket stream.

        The newly created socket wrapper takes ownership of the socket being passed in.
        The existing socket must already be connected.

        :param sock_or_fd: a socket object or file descriptor
        :return: a socket stream

        """
        sock = _validate_socket(sock_or_fd, socket.SOCK_STREAM, require_connected=True)
        return await get_async_backend().wrap_stream_socket(sock)


class UNIXSocketStream(SocketStream):
    @classmethod
    async def from_socket(cls, sock_or_fd: socket.socket | int) -> UNIXSocketStream:
        """
        Wrap an existing socket object or file descriptor as a UNIX socket stream.

        The newly created socket wrapper takes ownership of the socket being passed in.
        The existing socket must already be connected.

        :param sock_or_fd: a socket object or file descriptor
        :return: a UNIX socket stream

        """
        sock = _validate_socket(
            sock_or_fd, socket.SOCK_STREAM, socket.AF_UNIX, require_connected=True
        )
        return await get_async_backend().wrap_unix_stream_socket(sock)

    @abstractmethod
    async def send_fds(self, message: bytes, fds: Collection[int | IOBase]) -> None:
        """
        Send file descriptors along with a message to the peer.

        :param message: a non-empty bytestring
        :param fds: a collection of files (either numeric file descriptors or open file
            or socket objects)
        """

    @abstractmethod
    async def receive_fds(self, msglen: int, maxfds: int) -> tuple[bytes, list[int]]:
        """
        Receive file descriptors along with a message from the peer.

        :param msglen: length of the message to expect from the peer
        :param maxfds: maximum number of file descriptors to expect from the peer
        :return: a tuple of (message, file descriptors)
        """


class SocketListener(Listener[SocketStream], _SocketProvider):
    """
    Listens to incoming socket connections.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(
        cls,
        sock_or_fd: socket.socket | int,
    ) -> SocketListener:
        """
        Wrap an existing socket object or file descriptor as a socket listener.

        The newly created listener takes ownership of the socket being passed in.

        :param sock_or_fd: a socket object or file descriptor
        :return: a socket listener

        """
        sock = _validate_socket(sock_or_fd, socket.SOCK_STREAM, require_bound=True)
        return await get_async_backend().wrap_listener_socket(sock)

    @abstractmethod
    async def accept(self) -> SocketStream:
        """Accept an incoming connection."""

    async def serve(
        self,
        handler: Callable[[SocketStream], Any],
        task_group: TaskGroup | None = None,
    ) -> None:
        from .. import create_task_group

        async with AsyncExitStack() as stack:
            if task_group is None:
                task_group = await stack.enter_async_context(create_task_group())

            while True:
                stream = await self.accept()
                task_group.start_soon(handler, stream)


class UDPSocket(UnreliableObjectStream[UDPPacketType], _SocketProvider):
    """
    Represents an unconnected UDP socket.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(cls, sock_or_fd: socket.socket | int) -> UDPSocket:
        """
        Wrap an existing socket object or file descriptor as a UDP socket.

        The newly created socket wrapper takes ownership of the socket being passed in.
        The existing socket must be bound to a local address.

        :param sock_or_fd: a socket object or file descriptor
        :return: a UDP socket

        """
        sock = _validate_socket(sock_or_fd, socket.SOCK_DGRAM, require_bound=True)
        return await get_async_backend().wrap_udp_socket(sock)

    async def sendto(self, data: bytes, host: str, port: int) -> None:
        """
        Alias for :meth:`~.UnreliableObjectSendStream.send` ((data, (host, port))).

        """
        return await self.send((data, (host, port)))


class ConnectedUDPSocket(UnreliableObjectStream[bytes], _SocketProvider):
    """
    Represents an connected UDP socket.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(cls, sock_or_fd: socket.socket | int) -> ConnectedUDPSocket:
        """
        Wrap an existing socket object or file descriptor as a connected UDP socket.

        The newly created socket wrapper takes ownership of the socket being passed in.
        The existing socket must already be connected.

        :param sock_or_fd: a socket object or file descriptor
        :return: a connected UDP socket

        """
        sock = _validate_socket(
            sock_or_fd,
            socket.SOCK_DGRAM,
            require_connected=True,
        )
        return await get_async_backend().wrap_connected_udp_socket(sock)


class UNIXDatagramSocket(
    UnreliableObjectStream[UNIXDatagramPacketType], _SocketProvider
):
    """
    Represents an unconnected Unix datagram socket.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(
        cls,
        sock_or_fd: socket.socket | int,
    ) -> UNIXDatagramSocket:
        """
        Wrap an existing socket object or file descriptor as a UNIX datagram
        socket.

        The newly created socket wrapper takes ownership of the socket being passed in.

        :param sock_or_fd: a socket object or file descriptor
        :return: a UNIX datagram socket

        """
        sock = _validate_socket(sock_or_fd, socket.SOCK_DGRAM, socket.AF_UNIX)
        return await get_async_backend().wrap_unix_datagram_socket(sock)

    async def sendto(self, data: bytes, path: str) -> None:
        """Alias for :meth:`~.UnreliableObjectSendStream.send` ((data, path))."""
        return await self.send((data, path))


class ConnectedUNIXDatagramSocket(UnreliableObjectStream[bytes], _SocketProvider):
    """
    Represents a connected Unix datagram socket.

    Supports all relevant extra attributes from :class:`~SocketAttribute`.
    """

    @classmethod
    async def from_socket(
        cls,
        sock_or_fd: socket.socket | int,
    ) -> ConnectedUNIXDatagramSocket:
        """
        Wrap an existing socket object or file descriptor as a connected UNIX datagram
        socket.

        The newly created socket wrapper takes ownership of the socket being passed in.
        The existing socket must already be connected.

        :param sock_or_fd: a socket object or file descriptor
        :return: a connected UNIX datagram socket

        """
        sock = _validate_socket(
            sock_or_fd, socket.SOCK_DGRAM, socket.AF_UNIX, require_connected=True
        )
        return await get_async_backend().wrap_connected_unix_datagram_socket(sock)
<|endoftext|>from __future__ import annotations

from abc import abstractmethod
from signal import Signals
from typing import TYPE_CHECKING

from ._resources import AsyncResource

if TYPE_CHECKING:
    from ._streams import ByteReceiveStream, ByteSendStream


class Process(AsyncResource):
    """An asynchronous version of :class:`subprocess.Popen`."""

    @abstractmethod
    async def wait(self) -> int:
        """
        Wait until the process exits.

        :return: the exit code of the process
        """

    @abstractmethod
    def terminate(self) -> None:
        """
        Terminates the process, gracefully if possible.

        On Windows, this calls ``TerminateProcess()``.
        On POSIX systems, this sends ``SIGTERM`` to the process.

        .. seealso:: :meth:`subprocess.Popen.terminate`
        """

    @abstractmethod
    def kill(self) -> None:
        """
        Kills the process.

        On Windows, this calls ``TerminateProcess()``.
        On POSIX systems, this sends ``SIGKILL`` to the process.

        .. seealso:: :meth:`subprocess.Popen.kill`
        """

    @abstractmethod
    def send_signal(self, signal: Signals) -> None:
        """
        Send a signal to the subprocess.

        .. seealso:: :meth:`subprocess.Popen.send_signal`

        :param signal: the signal number (e.g. :data:`signal.SIGHUP`)
        """

    @property
    @abstractmethod
    def pid(self) -> int:
        """The process ID of the process."""

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        """
        The return code of the process. If the process has not yet terminated, this will
        be ``None``.
        """

    @property
    @abstractmethod
    def stdin(self) -> ByteSendStream | None:
        """The stream for the standard input of the process."""

    @property
    @abstractmethod
    def stdout(self) -> ByteReceiveStream | None:
        """The stream for the standard output of the process."""

    @property
    @abstractmethod
    def stderr(self) -> ByteReceiveStream | None:
        """The stream for the standard error output of the process."""
<|endoftext|>from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource

if TYPE_CHECKING:
    from ._tasks import TaskGroup

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class UnreliableObjectReceiveStream(
    AsyncResource, TypedAttributeProvider, Generic[T_co]
):
    """
    An interface for receiving objects.

    This interface makes no guarantees that the received messages arrive in the order in
    which they were sent, or that no messages are missed.

    Asynchronously iterating over objects of this type will yield objects matching the
    given type parameter.
    """

    def __aiter__(self) -> UnreliableObjectReceiveStream[T_co]:
        return self

    async def __anext__(self) -> T_co:
        try:
            return await self.receive()
        except EndOfStream:
            raise StopAsyncIteration from None

    @abstractmethod
    async def receive(self) -> T_co:
        """
        Receive the next item.

        :raises ~anyio.ClosedResourceError: if the receive stream has been explicitly
            closed
        :raises ~anyio.EndOfStream: if this stream has been closed from the other end
        :raises ~anyio.BrokenResourceError: if this stream has been rendered unusable
            due to external causes
        """


class UnreliableObjectSendStream(
    AsyncResource, TypedAttributeProvider, Generic[T_contra]
):
    """
    An interface for sending objects.

    This interface makes no guarantees that the messages sent will reach the
    recipient(s) in the same order in which they were sent, or at all.
    """

    @abstractmethod
    async def send(self, item: T_contra) -> None:
        """
        Send an item to the peer(s).

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if the send stream has been explicitly
            closed
        :raises ~anyio.BrokenResourceError: if this stream has been rendered unusable
            due to external causes
        """


class UnreliableObjectStream(
    UnreliableObjectReceiveStream[T_Item], UnreliableObjectSendStream[T_Item]
):
    """
    A bidirectional message stream which does not guarantee the order or reliability of
    message delivery.
    """


class ObjectReceiveStream(UnreliableObjectReceiveStream[T_co]):
    """
    A receive message stream which guarantees that messages are received in the same
    order in which they were sent, and that no messages are missed.
    """


class ObjectSendStream(UnreliableObjectSendStream[T_contra]):
    """
    A send message stream which guarantees that messages are delivered in the same order
    in which they were sent, without missing any messages in the middle.
    """


class ObjectStream(
    ObjectReceiveStream[T_Item],
    ObjectSendStream[T_Item],
    UnreliableObjectStream[T_Item],
):
    """
    A bidirectional message stream which guarantees the order and reliability of message
    delivery.
    """

    @abstractmethod
    async def send_eof(self) -> None:
        """
        Send an end-of-file indication to the peer.

        You should not try to send any further data to this stream after calling this
        method. This method is idempotent (does nothing on successive calls).
        """


class ByteReceiveStream(AsyncResource, TypedAttributeProvider):
    """
    An interface for receiving bytes from a single peer.

    Iterating this byte stream will yield a byte string of arbitrary length, but no more
    than 65536 bytes.
    """

    def __aiter__(self) -> ByteReceiveStream:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.receive()
        except EndOfStream:
            raise StopAsyncIteration from None

    @abstractmethod
    async def receive(self, max_bytes: int = 65536) -> bytes:
        """
        Receive at most ``max_bytes`` bytes from the peer.

        .. note:: Implementers of this interface should not return an empty
            :class:`bytes` object, and users should ignore them.

        :param max_bytes: maximum number of bytes to receive (must be a positive
            integer)
        :return: the received bytes
        :raises ValueError: if ``max_bytes`` is less than 1
        :raises ~anyio.EndOfStream: if this stream has been closed from the other end
        """


class ByteSendStream(AsyncResource, TypedAttributeProvider):
    """An interface for sending bytes to a single peer."""

    @abstractmethod
    async def send(self, item: bytes) -> None:
        """
        Send the given bytes to the peer.

        :param item: the bytes to send
        """


class ByteStream(ByteReceiveStream, ByteSendStream):
    """A bidirectional byte stream."""

    @abstractmethod
    async def send_eof(self) -> None:
        """
        Send an end-of-file indication to the peer.

        You should not try to send any further data to this stream after calling this
        method. This method is idempotent (does nothing on successive calls).
        """


#: Type alias for all unreliable bytes-oriented receive streams.
AnyUnreliableByteReceiveStream: TypeAlias = (
    UnreliableObjectReceiveStream[bytes] | ByteReceiveStream
)
#: Type alias for all unreliable bytes-oriented send streams.
AnyUnreliableByteSendStream: TypeAlias = (
    UnreliableObjectSendStream[bytes] | ByteSendStream
)
#: Type alias for all unreliable bytes-oriented streams.
AnyUnreliableByteStream: TypeAlias = UnreliableObjectStream[bytes] | ByteStream
#: Type alias for all bytes-oriented receive streams.
AnyByteReceiveStream: TypeAlias = ObjectReceiveStream[bytes] | ByteReceiveStream
#: Type alias for all bytes-oriented send streams.
AnyByteSendStream: TypeAlias = ObjectSendStream[bytes] | ByteSendStream
#: Type alias for all bytes-oriented streams.
AnyByteStream: TypeAlias = ObjectStream[bytes] | ByteStream


class Listener(AsyncResource, TypedAttributeProvider, Generic[T_co]):
    """An interface for objects that let you accept incoming connections."""

    @abstractmethod
    async def serve(
        self, handler: Callable[[T_co], Any], task_group: TaskGroup | None = None
    ) -> None:
        """
        Accept incoming connections as they come in and start tasks to handle them.

        :param handler: a callable that will be used to handle each accepted connection
        :param task_group: the task group that will be used to start tasks for handling
            each accepted connection (if omitted, an ad-hoc task group will be created)
        """


class ObjectStreamConnectable(Generic[T_co], metaclass=ABCMeta):
    @abstractmethod
    async def connect(self) -> ObjectStream[T_co]:
        """
        Connect to the remote endpoint.

        :return: an object stream connected to the remote end
        :raises ConnectionFailed: if the connection fails
        """


class ByteStreamConnectable(metaclass=ABCMeta):
    @abstractmethod
    async def connect(self) -> ByteStream:
        """
        Connect to the remote endpoint.

        :return: a bytestream connected to the remote end
        :raises ConnectionFailed: if the connection fails
        """


#: Type alias for all connectables returning bytestreams or bytes-oriented object streams
AnyByteStreamConnectable: TypeAlias = (
    ObjectStreamConnectable[bytes] | ByteStreamConnectable
)
<|endoftext|>from __future__ import annotations

import sys
import types
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_T = TypeVar("_T")


class TestRunner(metaclass=ABCMeta):
    """
    Encapsulates a running event loop. Every call made through this object will use the
    same event loop.
    """

    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None: ...

    @abstractmethod
    def run_asyncgen_fixture(
        self,
        fixture_func: Callable[..., AsyncGenerator[_T, Any]],
        kwargs: dict[str, Any],
    ) -> Iterable[_T]:
        """
        Run an async generator fixture.

        :param fixture_func: the fixture function
        :param kwargs: keyword arguments to call the fixture function with
        :return: an iterator yielding the value yielded from the async generator
        """

    @abstractmethod
    def run_fixture(
        self,
        fixture_func: Callable[..., Coroutine[Any, Any, _T]],
        kwargs: dict[str, Any],
    ) -> _T:
        """
        Run an async fixture.

        :param fixture_func: the fixture function
        :param kwargs: keyword arguments to call the fixture function with
        :return: the return value of the fixture function
        """

    @abstractmethod
    def run_test(
        self, test_func: Callable[..., Coroutine[Any, Any, Any]], kwargs: dict[str, Any]
    ) -> None:
        """
        Run an async test function.

        :param test_func: the test function
        :param kwargs: keyword arguments to call the test function with
        """

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the test runner is running.

        :return: ``True`` if the coroutine is currently being run, ``False`` otherwise.
        """
<|endoftext|>from __future__ import annotations

__all__ = (
    "FileReadStream",
    "FileStreamAttribute",
    "FileWriteStream",
)

from collections.abc import Callable, Mapping
from io import SEEK_SET, UnsupportedOperation
from os import PathLike
from pathlib import Path
from typing import IO, Any

from .. import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    TypedAttributeSet,
    to_thread,
    typed_attribute,
)
from ..abc import ByteReceiveStream, ByteSendStream


class FileStreamAttribute(TypedAttributeSet):
    #: the open file descriptor
    file: IO[bytes] = typed_attribute()
    #: the path of the file on the file system, if available (file must be a real file)
    path: Path = typed_attribute()
    #: the file number, if available (file must be a real file or a TTY)
    fileno: int = typed_attribute()


class _BaseFileStream:
    def __init__(self, file: IO[bytes]):
        self._file = file

    async def aclose(self) -> None:
        await to_thread.run_sync(self._file.close)

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        attributes: dict[Any, Callable[[], Any]] = {
            FileStreamAttribute.file: lambda: self._file,
        }

        if hasattr(self._file, "name"):
            attributes[FileStreamAttribute.path] = lambda: Path(self._file.name)

        try:
            self._file.fileno()
        except UnsupportedOperation:
            pass
        else:
            attributes[FileStreamAttribute.fileno] = lambda: self._file.fileno()

        return attributes


class FileReadStream(_BaseFileStream, ByteReceiveStream):
    """
    A byte stream that reads from a file in the file system.

    :param file: a file that has been opened for reading in binary mode

    .. versionadded:: 3.0
    """

    @classmethod
    async def from_path(cls, path: str | PathLike[str]) -> FileReadStream:
        """
        Create a file read stream by opening the given file.

        :param path: path of the file to read from

        """
        file = await to_thread.run_sync(Path(path).open, "rb")
        return cls(file)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        try:
            data = await to_thread.run_sync(self._file.read, max_bytes)
        except ValueError:
            raise ClosedResourceError from None
        except OSError as exc:
            raise BrokenResourceError from exc

        if data:
            return data
        else:
            raise EndOfStream

    async def seek(self, position: int, whence: int = SEEK_SET) -> int:
        """
        Seek the file to the given position.

        .. seealso:: :meth:`io.IOBase.seek`

        .. note:: Not all file descriptors are seekable.

        :param position: position to seek the file to
        :param whence: controls how ``position`` is interpreted
        :return: the new absolute position
        :raises OSError: if the file is not seekable

        """
        return await to_thread.run_sync(self._file.seek, position, whence)

    async def tell(self) -> int:
        """
        Return the current stream position.

        .. note:: Not all file descriptors are seekable.

        :return: the current absolute position
        :raises OSError: if the file is not seekable

        """
        return await to_thread.run_sync(self._file.tell)


class FileWriteStream(_BaseFileStream, ByteSendStream):
    """
    A byte stream that writes to a file in the file system.

    :param file: a file that has been opened for writing in binary mode

    .. versionadded:: 3.0
    """

    @classmethod
    async def from_path(
        cls, path: str | PathLike[str], append: bool = False
    ) -> FileWriteStream:
        """
        Create a file write stream by opening the given file for writing.

        :param path: path of the file to write to
        :param append: if ``True``, open the file for appending; if ``False``, any
            existing file at the given path will be truncated

        """
        mode = "ab" if append else "wb"
        file = await to_thread.run_sync(Path(path).open, mode)
        return cls(file)

    async def send(self, item: bytes) -> None:
        try:
            await to_thread.run_sync(self._file.write, item)
        except ValueError:
            raise ClosedResourceError from None
        except OSError as exc:
            raise BrokenResourceError from exc
<|endoftext|>from __future__ import annotations

__all__ = (
    "MemoryObjectReceiveStream",
    "MemoryObjectSendStream",
    "MemoryObjectStreamStatistics",
)

import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar

from .._core._exceptions import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from .._core._testing import TaskInfo, get_current_task
from ..abc import ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class MemoryObjectStreamStatistics(NamedTuple):
    current_buffer_used: int  #: number of items stored in the buffer
    #: maximum number of items that can be stored on this stream (or :data:`math.inf`)
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send stream
    open_receive_streams: int  #: number of unclosed clones of the receive stream
    #: number of tasks blocked on :meth:`MemoryObjectSendStream.send`
    tasks_waiting_send: int
    #: number of tasks blocked on :meth:`MemoryObjectReceiveStream.receive`
    tasks_waiting_receive: int


@dataclass(eq=False)
class _MemoryObjectItemReceiver(Generic[T_Item]):
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    item: T_Item = field(init=False)

    def __repr__(self) -> str:
        # When item is not defined, we get following error with default __repr__:
        # AttributeError: 'MemoryObjectItemReceiver' object has no attribute 'item'
        item = getattr(self, "item", None)
        return f"{self.__class__.__name__}(task_info={self.task_info}, item={item!r})"


@dataclass(eq=False)
class _MemoryObjectStreamState(Generic[T_Item]):
    max_buffer_size: float = field()
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    waiting_receivers: OrderedDict[Event, _MemoryObjectItemReceiver[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )

    def statistics(self) -> MemoryObjectStreamStatistics:
        return MemoryObjectStreamStatistics(
            len(self.buffer),
            self.max_buffer_size,
            self.open_send_channels,
            self.open_receive_channels,
            len(self.waiting_senders),
            len(self.waiting_receivers),
        )


@dataclass(eq=False)
class MemoryObjectReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    _state: _MemoryObjectStreamState[T_co]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def receive_nowait(self) -> T_co:
        """
        Receive the next item if it can be done without waiting.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and this stream has been
            closed from the sending end
        :raises ~anyio.WouldBlock: if there are no items in the buffer and no tasks
            waiting to send

        """
        if self._closed:
            raise ClosedResourceError

        if self._state.waiting_senders:
            # Get the item from the next sender
            send_event, item = self._state.waiting_senders.popitem(last=False)
            self._state.buffer.append(item)
            send_event.set()

        if self._state.buffer:
            return self._state.buffer.popleft()
        elif not self._state.open_send_channels:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            # Add ourselves in the queue
            receive_event = Event()
            receiver = _MemoryObjectItemReceiver[T_co]()
            self._state.waiting_receivers[receive_event] = receiver

            try:
                await receive_event.wait()
            finally:
                self._state.waiting_receivers.pop(receive_event, None)

            try:
                return receiver.item
            except AttributeError:
                raise EndOfStream from None

    def clone(self) -> MemoryObjectReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the receiving end of the memory stream be considered closed by the sending ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectReceiveStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_receive_channels -= 1
            if self._state.open_receive_channels == 0:
                send_events = list(self._state.waiting_senders.keys())
                for event in send_events:
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectReceiveStream[T_co]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


@dataclass(eq=False)
class MemoryObjectSendStream(ObjectSendStream[T_contra], Generic[T_contra]):
    _state: _MemoryObjectStreamState[T_contra]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send an item immediately if it can be done without waiting.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end
        :raises ~anyio.WouldBlock: if the buffer is full and there are no tasks waiting
            to receive

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.open_receive_channels:
            raise BrokenResourceError

        while self._state.waiting_receivers:
            receive_event, receiver = self._state.waiting_receivers.popitem(last=False)
            if not receiver.task_info.has_pending_cancellation():
                receiver.item = item
                receive_event.set()
                return

        if len(self._state.buffer) < self._state.max_buffer_size:
            self._state.buffer.append(item)
        else:
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send an item to the stream.

        If the buffer is full, this method blocks until there is again room in the
        buffer or the item can be sent directly to a receiver.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end

        """
        await checkpoint()
        try:
            self.send_nowait(item)
        except WouldBlock:
            # Wait until there's someone on the receiving end
            send_event = Event()
            self._state.waiting_senders[send_event] = item
            try:
                await send_event.wait()
            except BaseException:
                self._state.waiting_senders.pop(send_event, None)
                raise

            if send_event in self._state.waiting_senders:
                del self._state.waiting_senders[send_event]
                raise BrokenResourceError from None

    def clone(self) -> MemoryObjectSendStream[T_contra]:
        """
        Create a clone of this send stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the sending end of the memory stream be considered closed by the receiving ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                receive_events = list(self._state.waiting_receivers.keys())
                self._state.waiting_receivers.clear()
                for event in receive_events:
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectSendStream[T_contra]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )
<|endoftext|>from __future__ import annotations

__all__ = (
    "TextConnectable",
    "TextReceiveStream",
    "TextSendStream",
    "TextStream",
)

import codecs
import sys
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

from ..abc import (
    AnyByteReceiveStream,
    AnyByteSendStream,
    AnyByteStream,
    AnyByteStreamConnectable,
    ObjectReceiveStream,
    ObjectSendStream,
    ObjectStream,
    ObjectStreamConnectable,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


@dataclass(eq=False)
class TextReceiveStream(ObjectReceiveStream[str]):
    """
    Stream wrapper that decodes bytes to strings using the given encoding.

    Decoding is done using :class:`~codecs.IncrementalDecoder` which returns any
    completely received unicode characters as soon as they come in.

    :param transport_stream: any bytes-based receive stream
    :param encoding: character encoding to use for decoding bytes to strings (defaults
        to ``utf-8``)
    :param errors: handling scheme for decoding errors (defaults to ``strict``; see the
        `codecs module documentation`_ for a comprehensive list of options)

    .. _codecs module documentation:
        https://docs.python.org/3/library/codecs.html#codec-objects
    """

    transport_stream: AnyByteReceiveStream
    encoding: InitVar[str] = "utf-8"
    errors: InitVar[str] = "strict"
    _decoder: codecs.IncrementalDecoder = field(init=False)

    def __post_init__(self, encoding: str, errors: str) -> None:
        decoder_class = codecs.getincrementaldecoder(encoding)
        self._decoder = decoder_class(errors=errors)

    async def receive(self) -> str:
        while True:
            chunk = await self.transport_stream.receive()
            decoded = self._decoder.decode(chunk)
            if decoded:
                return decoded

    async def aclose(self) -> None:
        await self.transport_stream.aclose()
        self._decoder.reset()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.transport_stream.extra_attributes


@dataclass(eq=False)
class TextSendStream(ObjectSendStream[str]):
    """
    Sends strings to the wrapped stream as bytes using the given encoding.

    :param AnyByteSendStream transport_stream: any bytes-based send stream
    :param str encoding: character encoding to use for encoding strings to bytes
        (defaults to ``utf-8``)
    :param str errors: handling scheme for encoding errors (defaults to ``strict``; see
        the `codecs module documentation`_ for a comprehensive list of options)

    .. _codecs module documentation:
        https://docs.python.org/3/library/codecs.html#codec-objects
    """

    transport_stream: AnyByteSendStream
    encoding: InitVar[str] = "utf-8"
    errors: str = "strict"
    _encoder: Callable[..., tuple[bytes, int]] = field(init=False)

    def __post_init__(self, encoding: str) -> None:
        self._encoder = codecs.getencoder(encoding)

    async def send(self, item: str) -> None:
        encoded = self._encoder(item, self.errors)[0]
        await self.transport_stream.send(encoded)

    async def aclose(self) -> None:
        await self.transport_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.transport_stream.extra_attributes


@dataclass(eq=False)
class TextStream(ObjectStream[str]):
    """
    A bidirectional stream that decodes bytes to strings on receive and encodes strings
    to bytes on send.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param AnyByteStream transport_stream: any bytes-based stream
    :param str encoding: character encoding to use for encoding/decoding strings to/from
        bytes (defaults to ``utf-8``)
    :param str errors: handling scheme for encoding errors (defaults to ``strict``; see
        the `codecs module documentation`_ for a comprehensive list of options)

    .. _codecs module documentation:
        https://docs.python.org/3/library/codecs.html#codec-objects
    """

    transport_stream: AnyByteStream
    encoding: InitVar[str] = "utf-8"
    errors: InitVar[str] = "strict"
    _receive_stream: TextReceiveStream = field(init=False)
    _send_stream: TextSendStream = field(init=False)

    def __post_init__(self, encoding: str, errors: str) -> None:
        self._receive_stream = TextReceiveStream(
            self.transport_stream, encoding=encoding, errors=errors
        )
        self._send_stream = TextSendStream(
            self.transport_stream, encoding=encoding, errors=errors
        )

    async def receive(self) -> str:
        return await self._receive_stream.receive()

    async def send(self, item: str) -> None:
        await self._send_stream.send(item)

    async def send_eof(self) -> None:
        await self.transport_stream.send_eof()

    async def aclose(self) -> None:
        await self._send_stream.aclose()
        await self._receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self._send_stream.extra_attributes,
            **self._receive_stream.extra_attributes,
        }


class TextConnectable(ObjectStreamConnectable[str]):
    def __init__(self, connectable: AnyByteStreamConnectable):
        """
        :param connectable: the bytestream endpoint to wrap

        """
        self.connectable = connectable

    @override
    async def connect(self) -> TextStream:
        stream = await self.connectable.connect()
        return TextStream(stream)
<|endoftext|>from __future__ import annotations

__all__ = (
    "TLSAttribute",
    "TLSConnectable",
    "TLSListener",
    "TLSStream",
)

import logging
import re
import ssl
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from ssl import SSLContext
from typing import Any, TypeAlias, TypeVar

from .. import (
    BrokenResourceError,
    EndOfStream,
    aclose_forcefully,
    get_cancelled_exc_class,
    to_thread,
)
from .._core._typedattr import TypedAttributeSet, typed_attribute
from ..abc import (
    AnyByteStream,
    AnyByteStreamConnectable,
    ByteStream,
    ByteStreamConnectable,
    Listener,
    TaskGroup,
)

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

T_Retval = TypeVar("T_Retval")
PosArgsT = TypeVarTuple("PosArgsT")
_PCTRTT: TypeAlias = tuple[tuple[str, str], ...]
_PCTRTTT: TypeAlias = tuple[_PCTRTT, ...]


class TLSAttribute(TypedAttributeSet):
    """Contains Transport Layer Security related attributes."""

    #: the selected ALPN protocol
    alpn_protocol: str | None = typed_attribute()
    #: the channel binding for type ``tls-unique``
    channel_binding_tls_unique: bytes = typed_attribute()
    #: the selected cipher
    cipher: tuple[str, str, int] = typed_attribute()
    #: the peer certificate in dictionary form (see :meth:`ssl.SSLSocket.getpeercert`
    # for more information)
    peer_certificate: None | (dict[str, str | _PCTRTTT | _PCTRTT]) = typed_attribute()
    #: the peer certificate in binary form
    peer_certificate_binary: bytes | None = typed_attribute()
    #: ``True`` if this is the server side of the connection
    server_side: bool = typed_attribute()
    #: ciphers shared by the client during the TLS handshake (``None`` if this is the
    #: client side)
    shared_ciphers: list[tuple[str, str, int]] | None = typed_attribute()
    #: the :class:`~ssl.SSLObject` used for encryption
    ssl_object: ssl.SSLObject = typed_attribute()
    #: ``True`` if this stream does (and expects) a closing TLS handshake when the
    #: stream is being closed
    standard_compatible: bool = typed_attribute()
    #: the TLS protocol version (e.g. ``TLSv1.2``)
    tls_version: str = typed_attribute()


@dataclass(eq=False)
class TLSStream(ByteStream):
    """
    A stream wrapper that encrypts all sent data and decrypts received data.

    This class has no public initializer; use :meth:`wrap` instead.
    All extra attributes from :class:`~TLSAttribute` are supported.

    :var AnyByteStream transport_stream: the wrapped stream

    """

    transport_stream: AnyByteStream
    standard_compatible: bool
    _ssl_object: ssl.SSLObject
    _read_bio: ssl.MemoryBIO
    _write_bio: ssl.MemoryBIO

    @classmethod
    async def wrap(
        cls,
        transport_stream: AnyByteStream,
        *,
        server_side: bool | None = None,
        hostname: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        standard_compatible: bool = True,
    ) -> TLSStream:
        """
        Wrap an existing stream with Transport Layer Security.

        This performs a TLS handshake with the peer.

        :param transport_stream: a bytes-transporting stream to wrap
        :param server_side: ``True`` if this is the server side of the connection,
            ``False`` if this is the client side (if omitted, will be set to ``False``
            if ``hostname`` has been provided, ``False`` otherwise). Used only to create
            a default context when an explicit context has not been provided.
        :param hostname: host name of the peer (if host name checking is desired)
        :param ssl_context: the SSLContext object to use (if not provided, a secure
            default will be created)
        :param standard_compatible: if ``False``, skip the closing handshake when
            closing the connection, and don't raise an exception if the peer does the
            same
        :raises ~ssl.SSLError: if the TLS handshake fails

        """
        if server_side is None:
            server_side = not hostname

        if not ssl_context:
            purpose = (
                ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
            )
            ssl_context = ssl.create_default_context(purpose)

            # Re-enable detection of unexpected EOFs if it was disabled by Python
            if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
                ssl_context.options &= ~ssl.OP_IGNORE_UNEXPECTED_EOF

        bio_in = ssl.MemoryBIO()
        bio_out = ssl.MemoryBIO()

        # Resolve international host names using IDNA 2008.
        # Otherwise wrap_bio() would resolve them with IDNA 2003.
        if hostname is not None:
            from .._core._sockets import idna2008_resolve

            server_hostname: bytes | None = idna2008_resolve(hostname)
        else:
            server_hostname = None

        # External SSLContext implementations may do blocking I/O in wrap_bio(),
        # but the standard library implementation won't
        if type(ssl_context) is ssl.SSLContext:
            ssl_object = ssl_context.wrap_bio(
                bio_in,
                bio_out,
                server_side=server_side,
                server_hostname=server_hostname,
            )
        else:
            ssl_object = await to_thread.run_sync(
                ssl_context.wrap_bio,
                bio_in,
                bio_out,
                server_side,
                server_hostname,
                None,
            )

        wrapper = cls(
            transport_stream=transport_stream,
            standard_compatible=standard_compatible,
            _ssl_object=ssl_object,
            _read_bio=bio_in,
            _write_bio=bio_out,
        )
        await wrapper._call_sslobject_method(ssl_object.do_handshake)
        return wrapper

    async def _call_sslobject_method(
        self, func: Callable[[Unpack[PosArgsT]], T_Retval], *args: Unpack[PosArgsT]
    ) -> T_Retval:
        while True:
            try:
                result = func(*args)
            except ssl.SSLWantReadError:
                try:
                    # Flush any pending writes first
                    if self._write_bio.pending:
                        await self.transport_stream.send(self._write_bio.read())

                    data = await self.transport_stream.receive()
                except EndOfStream:
                    self._read_bio.write_eof()
                except OSError as exc:
                    self._read_bio.write_eof()
                    self._write_bio.write_eof()
                    raise BrokenResourceError from exc
                else:
                    self._read_bio.write(data)
            except ssl.SSLWantWriteError:
                await self.transport_stream.send(self._write_bio.read())
            except ssl.SSLSyscallError as exc:
                self._read_bio.write_eof()
                self._write_bio.write_eof()
                raise BrokenResourceError from exc
            except ssl.SSLError as exc:
                self._read_bio.write_eof()
                self._write_bio.write_eof()
                if isinstance(exc, ssl.SSLEOFError) or (
                    exc.strerror and "UNEXPECTED_EOF_WHILE_READING" in exc.strerror
                ):
                    if self.standard_compatible:
                        raise BrokenResourceError from exc
                    else:
                        raise EndOfStream from None

                raise
            else:
                # Flush any pending writes first
                if self._write_bio.pending:
                    await self.transport_stream.send(self._write_bio.read())

                return result

    async def unwrap(self) -> tuple[AnyByteStream, bytes]:
        """
        Does the TLS closing handshake.

        :return: a tuple of (wrapped byte stream, bytes left in the read buffer)

        """
        await self._call_sslobject_method(self._ssl_object.unwrap)
        self._read_bio.write_eof()
        self._write_bio.write_eof()
        return self.transport_stream, self._read_bio.read()

    async def aclose(self) -> None:
        if self.standard_compatible:
            try:
                await self.unwrap()
            except BaseException:
                await aclose_forcefully(self.transport_stream)
                raise

        await self.transport_stream.aclose()

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        data = await self._call_sslobject_method(self._ssl_object.read, max_bytes)
        if not data:
            raise EndOfStream

        return data

    async def send(self, item: bytes) -> None:
        await self._call_sslobject_method(self._ssl_object.write, item)

    async def send_eof(self) -> None:
        tls_version = self.extra(TLSAttribute.tls_version)
        match = re.match(r"TLSv(\d+)(?:\.(\d+))?", tls_version)
        if match:
            major, minor = int(match.group(1)), int(match.group(2) or 0)
            if (major, minor) < (1, 3):
                raise NotImplementedError(
                    f"send_eof() requires at least TLSv1.3; current "
                    f"session uses {tls_version}"
                )

        raise NotImplementedError(
            "send_eof() has not yet been implemented for TLS streams"
        )

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self.transport_stream.extra_attributes,
            TLSAttribute.alpn_protocol: self._ssl_object.selected_alpn_protocol,
            TLSAttribute.channel_binding_tls_unique: (
                self._ssl_object.get_channel_binding
            ),
            TLSAttribute.cipher: self._ssl_object.cipher,
            TLSAttribute.peer_certificate: lambda: self._ssl_object.getpeercert(False),
            TLSAttribute.peer_certificate_binary: lambda: self._ssl_object.getpeercert(
                True
            ),
            TLSAttribute.server_side: lambda: self._ssl_object.server_side,
            TLSAttribute.shared_ciphers: lambda: (
                self._ssl_object.shared_ciphers()
                if self._ssl_object.server_side
                else None
            ),
            TLSAttribute.standard_compatible: lambda: self.standard_compatible,
            TLSAttribute.ssl_object: lambda: self._ssl_object,
            TLSAttribute.tls_version: self._ssl_object.version,
        }


@dataclass(eq=False)
class TLSListener(Listener[TLSStream]):
    """
    A convenience listener that wraps another listener and auto-negotiates a TLS session
    on every accepted connection.

    If the TLS handshake times out or raises an exception,
    :meth:`handle_handshake_error` is called to do whatever post-mortem processing is
    deemed necessary.

    Supports only the :attr:`~TLSAttribute.standard_compatible` extra attribute.

    :param Listener listener: the listener to wrap
    :param ssl_context: the SSL context object
    :param standard_compatible: a flag passed through to :meth:`TLSStream.wrap`
    :param handshake_timeout: time limit for the TLS handshake
        (passed to :func:`~anyio.fail_after`)
    """

    listener: Listener[Any]
    ssl_context: ssl.SSLContext
    standard_compatible: bool = True
    handshake_timeout: float = 30

    @staticmethod
    async def handle_handshake_error(exc: BaseException, stream: AnyByteStream) -> None:
        """
        Handle an exception raised during the TLS handshake.

        This method does 3 things:

        #. Forcefully closes the original stream
        #. Logs the exception (unless it was a cancellation exception) using the
           ``anyio.streams.tls`` logger
        #. Reraises the exception if it was a base exception or a cancellation exception

        :param exc: the exception
        :param stream: the original stream

        """
        await aclose_forcefully(stream)

        # Log all except cancellation exceptions
        if not isinstance(exc, get_cancelled_exc_class()):
            # CPython (as of 3.11.5) returns incorrect `sys.exc_info()` here when using
            # any asyncio implementation, so we explicitly pass the exception to log
            # (https://github.com/python/cpython/issues/108668). Trio does not have this
            # issue because it works around the CPython bug.
            logging.getLogger(__name__).exception(
                "Error during TLS handshake", exc_info=exc
            )

        # Only reraise base exceptions and cancellation exceptions
        if not isinstance(exc, Exception) or isinstance(exc, get_cancelled_exc_class()):
            raise  # noqa: PLE0704

    async def serve(
        self,
        handler: Callable[[TLSStream], Any],
        task_group: TaskGroup | None = None,
    ) -> None:
        @wraps(handler)
        async def handler_wrapper(stream: AnyByteStream) -> None:
            from .. import fail_after

            try:
                with fail_after(self.handshake_timeout):
                    wrapped_stream = await TLSStream.wrap(
                        stream,
                        ssl_context=self.ssl_context,
                        standard_compatible=self.standard_compatible,
                    )
            except BaseException as exc:
                await self.handle_handshake_error(exc, stream)
            else:
                await handler(wrapped_stream)

        await self.listener.serve(handler_wrapper, task_group)

    async def aclose(self) -> None:
        await self.listener.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            TLSAttribute.standard_compatible: lambda: self.standard_compatible,
        }


class TLSConnectable(ByteStreamConnectable):
    """
    Wraps another connectable and does TLS negotiation after a successful connection.

    :param connectable: the connectable to wrap
    :param hostname: host name of the server (if host name checking is desired)
    :param ssl_context: the SSLContext object to use (if not provided, a secure default
        will be created)
    :param standard_compatible: if ``False``, skip the closing handshake when closing
        the connection, and don't raise an exception if the server does the same
    """

    def __init__(
        self,
        connectable: AnyByteStreamConnectable,
        *,
        hostname: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        standard_compatible: bool = True,
    ) -> None:
        self.connectable = connectable
        self.ssl_context: SSLContext = ssl_context or ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH
        )
        if not isinstance(self.ssl_context, ssl.SSLContext):
            raise TypeError(
                "ssl_context must be an instance of ssl.SSLContext, not "
                f"{type(self.ssl_context).__name__}"
            )
        self.hostname = hostname
        self.standard_compatible = standard_compatible

    @override
    async def connect(self) -> TLSStream:
        stream = await self.connectable.connect()
        try:
            return await TLSStream.wrap(
                stream,
                hostname=self.hostname,
                ssl_context=self.ssl_context,
                standard_compatible=self.standard_compatible,
            )
        except BaseException:
            await aclose_forcefully(stream)
            raise
<|endoftext|>from __future__ import annotations

__all__ = (
    "BufferedByteReceiveStream",
    "BufferedByteStream",
    "BufferedConnectable",
)

import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, SupportsIndex

from .. import ClosedResourceError, DelimiterNotFound, EndOfStream, IncompleteRead
from ..abc import (
    AnyByteReceiveStream,
    AnyByteStream,
    AnyByteStreamConnectable,
    ByteReceiveStream,
    ByteStream,
    ByteStreamConnectable,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


@dataclass(eq=False)
class BufferedByteReceiveStream(ByteReceiveStream):
    """
    Wraps any bytes-based receive stream and uses a buffer to provide sophisticated
    receiving capabilities in the form of a byte stream.
    """

    receive_stream: AnyByteReceiveStream
    _buffer: bytearray = field(init=False, default_factory=bytearray)
    _closed: bool = field(init=False, default=False)

    async def aclose(self) -> None:
        await self.receive_stream.aclose()
        self._closed = True

    @property
    def buffer(self) -> bytes:
        """The bytes currently in the buffer."""
        return bytes(self._buffer)

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.receive_stream.extra_attributes

    def feed_data(self, data: Iterable[SupportsIndex], /) -> None:
        """
        Append data directly into the buffer.

        Any data in the buffer will be consumed by receive operations before receiving
        anything from the wrapped stream.

        :param data: the data to append to the buffer (can be bytes or anything else
            that supports ``__index__()``)

        """
        self._buffer.extend(data)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        if self._closed:
            raise ClosedResourceError

        if self._buffer:
            chunk = bytes(self._buffer[:max_bytes])
            del self._buffer[:max_bytes]
            return chunk
        elif isinstance(self.receive_stream, ByteReceiveStream):
            return await self.receive_stream.receive(max_bytes)
        else:
            # With a bytes-oriented object stream, we need to handle any surplus bytes
            # we get from the receive() call
            chunk = await self.receive_stream.receive()
            if len(chunk) > max_bytes:
                # Save the surplus bytes in the buffer
                self._buffer.extend(chunk[max_bytes:])
                return chunk[:max_bytes]
            else:
                return chunk

    async def receive_exactly(self, nbytes: int) -> bytes:
        """
        Read exactly the given amount of bytes from the stream.

        :param nbytes: the number of bytes to read
        :return: the bytes read
        :raises ~anyio.IncompleteRead: if the stream was closed before the requested
            amount of bytes could be read from the stream

        """
        while True:
            remaining = nbytes - len(self._buffer)
            if remaining <= 0:
                retval = self._buffer[:nbytes]
                del self._buffer[:nbytes]
                return bytes(retval)

            try:
                if isinstance(self.receive_stream, ByteReceiveStream):
                    chunk = await self.receive_stream.receive(remaining)
                else:
                    chunk = await self.receive_stream.receive()
            except EndOfStream as exc:
                raise IncompleteRead from exc

            self._buffer.extend(chunk)

    async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
        """
        Read from the stream until the delimiter is found or max_bytes have been read.

        :param delimiter: the marker to look for in the stream
        :param max_bytes: maximum number of bytes that will be read before raising
            :exc:`~anyio.DelimiterNotFound`
        :return: the bytes read (not including the delimiter)
        :raises ~anyio.IncompleteRead: if the stream was closed before the delimiter
            was found
        :raises ~anyio.DelimiterNotFound: if the delimiter is not found within the
            bytes read up to the maximum allowed

        """
        delimiter_size = len(delimiter)
        offset = 0
        while True:
            # Check if the delimiter can be found in the current buffer
            index = self._buffer.find(delimiter, offset)
            if index >= 0:
                found = self._buffer[:index]
                del self._buffer[: index + len(delimiter) :]
                return bytes(found)

            # Check if the buffer is already at or over the limit
            if len(self._buffer) >= max_bytes:
                raise DelimiterNotFound(max_bytes)

            # Read more data into the buffer from the socket
            try:
                data = await self.receive_stream.receive()
            except EndOfStream as exc:
                raise IncompleteRead from exc

            # Move the offset forward and add the new data to the buffer
            offset = max(len(self._buffer) - delimiter_size + 1, 0)
            self._buffer.extend(data)


class BufferedByteStream(BufferedByteReceiveStream, ByteStream):
    """
    A full-duplex variant of :class:`BufferedByteReceiveStream`. All writes are passed
    through to the wrapped stream as-is.
    """

    def __init__(self, stream: AnyByteStream):
        """
        :param stream: the stream to be wrapped

        """
        super().__init__(stream)
        self._stream = stream

    @override
    async def send_eof(self) -> None:
        await self._stream.send_eof()

    @override
    async def send(self, item: bytes) -> None:
        await self._stream.send(item)


class BufferedConnectable(ByteStreamConnectable):
    """
    Wraps a byte stream connectable to produce :class:`BufferedByteStream` connections.

    Use this when you want the streams returned by :meth:`connect` to have the buffered
    receive API (e.g. :meth:`~BufferedByteReceiveStream.receive_exactly` and
    :meth:`~BufferedByteReceiveStream.receive_until`).

    :param connectable: the byte stream connectable to wrap
    """

    def __init__(self, connectable: AnyByteStreamConnectable):
        """
        :param connectable: the connectable to wrap

        """
        self.connectable = connectable

    @override
    async def connect(self) -> BufferedByteStream:
        stream = await self.connectable.connect()
        return BufferedByteStream(stream)
<|endoftext|>from __future__ import annotations

__all__ = (
    "MultiListener",
    "StapledByteStream",
    "StapledObjectStream",
)

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..abc import (
    ByteReceiveStream,
    ByteSendStream,
    ByteStream,
    Listener,
    ObjectReceiveStream,
    ObjectSendStream,
    ObjectStream,
    TaskGroup,
)

T_Item = TypeVar("T_Item")
T_Stream = TypeVar("T_Stream")


@dataclass(eq=False)
class StapledByteStream(ByteStream):
    """
    Combines two byte streams into a single, bidirectional byte stream.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ByteSendStream send_stream: the sending byte stream
    :param ByteReceiveStream receive_stream: the receiving byte stream
    """

    send_stream: ByteSendStream
    receive_stream: ByteReceiveStream

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        return await self.receive_stream.receive(max_bytes)

    async def send(self, item: bytes) -> None:
        await self.send_stream.send(item)

    async def send_eof(self) -> None:
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        await self.send_stream.aclose()
        await self.receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self.send_stream.extra_attributes,
            **self.receive_stream.extra_attributes,
        }


@dataclass(eq=False)
class StapledObjectStream(ObjectStream[T_Item], Generic[T_Item]):
    """
    Combines two object streams into a single, bidirectional object stream.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ObjectSendStream send_stream: the sending object stream
    :param ObjectReceiveStream receive_stream: the receiving object stream
    """

    send_stream: ObjectSendStream[T_Item]
    receive_stream: ObjectReceiveStream[T_Item]

    async def receive(self) -> T_Item:
        return await self.receive_stream.receive()

    async def send(self, item: T_Item) -> None:
        await self.send_stream.send(item)

    def send_nowait(self, item: T_Item) -> None:
        try:
            send_nowait = self.send_stream.send_nowait  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise NotImplementedError(
                f"'send_nowait' method not implemented in {type(self.send_stream)}"
            ) from exc

        send_nowait(item)

    async def send_eof(self) -> None:
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        await self.send_stream.aclose()
        await self.receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self.send_stream.extra_attributes,
            **self.receive_stream.extra_attributes,
        }


@dataclass(eq=False)
class MultiListener(Listener[T_Stream], Generic[T_Stream]):
    """
    Combines multiple listeners into one, serving connections from all of them at once.

    Any MultiListeners in the given collection of listeners will have their listeners
    moved into this one.

    Extra attributes are provided from each listener, with each successive listener
    overriding any conflicting attributes from the previous one.

    :param listeners: listeners to serve
    :type listeners: Sequence[Listener[T_Stream]]
    """

    listeners: Sequence[Listener[T_Stream]]

    def __post_init__(self) -> None:
        listeners: list[Listener[T_Stream]] = []
        for listener in self.listeners:
            if isinstance(listener, MultiListener):
                listeners.extend(listener.listeners)
                del listener.listeners[:]  # type: ignore[attr-defined]
            else:
                listeners.append(listener)

        self.listeners = listeners

    async def serve(
        self, handler: Callable[[T_Stream], Any], task_group: TaskGroup | None = None
    ) -> None:
        from .. import create_task_group

        async with create_task_group() as tg:
            for listener in self.listeners:
                tg.start_soon(listener.serve, handler, task_group)

    async def aclose(self) -> None:
        for listener in self.listeners:
            await listener.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        attributes: dict = {}
        for listener in self.listeners:
            attributes.update(listener.extra_attributes)

        return attributes
<|endoftext|>from __future__ import annotations

from ..abc import AsyncResource  # noqa: TC001
from ._tasks import CancelScope


async def aclose_forcefully(resource: AsyncResource) -> None:
    """
    Close an asynchronous resource in a cancelled scope.

    Doing this closes the resource without waiting on anything.

    :param resource: the resource to close

    """
    with CancelScope() as scope:
        scope.cancel()
        await resource.aclose()
<|endoftext|>from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    ClassVar,
    Final,
    Generic,
    TypeVar,
    overload,
)

from .. import to_thread
from ..abc import AsyncResource
from ._synchronization import CapacityLimiter

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 14):
    from pathlib.types import PathInfo

if TYPE_CHECKING:
    from types import ModuleType

    from _typeshed import OpenBinaryMode, OpenTextMode, ReadableBuffer, WriteableBuffer
else:
    ReadableBuffer = OpenBinaryMode = OpenTextMode = WriteableBuffer = object


T = TypeVar("T", bound="Path")


class AsyncFile(AsyncResource, Generic[AnyStr]):
    """
    An asynchronous file object.

    This class wraps a standard file object and provides async friendly versions of the
    following blocking methods (where available on the original file object):

    * read
    * read1
    * readline
    * readlines
    * readinto
    * readinto1
    * write
    * writelines
    * truncate
    * seek
    * tell
    * flush

    All other methods are directly passed through.

    This class supports the asynchronous context manager protocol which closes the
    underlying file at the end of the context block.

    This class also supports asynchronous iteration::

        async with await open_file(...) as f:
            async for line in f:
                print(line)
    """

    def __init__(
        self, fp: IO[AnyStr], *, limiter: CapacityLimiter | None = None
    ) -> None:
        if limiter is not None and not isinstance(limiter, CapacityLimiter):
            raise TypeError(
                f"limiter must be a CapacityLimiter or None, not "
                f"{limiter.__class__.__name__}"
            )

        self._fp: Any = fp
        self._limiter = limiter

    def __getattr__(self, name: str) -> object:
        return getattr(self._fp, name)

    @property
    def limiter(self) -> CapacityLimiter | None:
        """The capacity limiter used by this file object, if not the global limiter."""
        return self._limiter

    @property
    def wrapped(self) -> IO[AnyStr]:
        """The wrapped file object."""
        return self._fp

    async def __aiter__(self) -> AsyncIterator[AnyStr]:
        while True:
            line = await self.readline()
            if line:
                yield line
            else:
                break

    async def aclose(self) -> None:
        return await to_thread.run_sync(self._fp.close, limiter=self._limiter)

    async def read(self, size: int = -1) -> AnyStr:
        return await to_thread.run_sync(self._fp.read, size, limiter=self._limiter)

    async def read1(self: AsyncFile[bytes], size: int = -1) -> bytes:
        return await to_thread.run_sync(self._fp.read1, size, limiter=self._limiter)

    async def readline(self) -> AnyStr:
        return await to_thread.run_sync(self._fp.readline, limiter=self._limiter)

    async def readlines(self) -> list[AnyStr]:
        return await to_thread.run_sync(self._fp.readlines, limiter=self._limiter)

    async def readinto(self: AsyncFile[bytes], b: WriteableBuffer) -> int:
        return await to_thread.run_sync(self._fp.readinto, b, limiter=self._limiter)

    async def readinto1(self: AsyncFile[bytes], b: WriteableBuffer) -> int:
        return await to_thread.run_sync(self._fp.readinto1, b, limiter=self._limiter)

    @overload
    async def write(self: AsyncFile[bytes], b: ReadableBuffer) -> int: ...

    @overload
    async def write(self: AsyncFile[str], b: str) -> int: ...

    async def write(self, b: ReadableBuffer | str) -> int:
        return await to_thread.run_sync(self._fp.write, b, limiter=self._limiter)

    @overload
    async def writelines(
        self: AsyncFile[bytes], lines: Iterable[ReadableBuffer]
    ) -> None: ...

    @overload
    async def writelines(self: AsyncFile[str], lines: Iterable[str]) -> None: ...

    async def writelines(self, lines: Iterable[ReadableBuffer] | Iterable[str]) -> None:
        return await to_thread.run_sync(
            self._fp.writelines, lines, limiter=self._limiter
        )

    async def truncate(self, size: int | None = None) -> int:
        return await to_thread.run_sync(self._fp.truncate, size, limiter=self._limiter)

    async def seek(self, offset: int, whence: int | None = os.SEEK_SET) -> int:
        return await to_thread.run_sync(
            self._fp.seek, offset, whence, limiter=self._limiter
        )

    async def tell(self) -> int:
        return await to_thread.run_sync(self._fp.tell, limiter=self._limiter)

    async def flush(self) -> None:
        return await to_thread.run_sync(self._fp.flush, limiter=self._limiter)


@overload
async def open_file(
    file: str | PathLike[str] | int,
    mode: OpenBinaryMode,
    buffering: int = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    closefd: bool = ...,
    opener: Callable[[str, int], int] | None = ...,
    *,
    limiter: CapacityLimiter | None = ...,
) -> AsyncFile[bytes]: ...


@overload
async def open_file(
    file: str | PathLike[str] | int,
    mode: OpenTextMode = ...,
    buffering: int = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    closefd: bool = ...,
    opener: Callable[[str, int], int] | None = ...,
    *,
    limiter: CapacityLimiter | None = ...,
) -> AsyncFile[str]: ...


async def open_file(
    file: str | PathLike[str] | int,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Callable[[str, int], int] | None = None,
    *,
    limiter: CapacityLimiter | None = None,
) -> AsyncFile[Any]:
    """
    Open a file asynchronously.

    Except for ``limiter``, the arguments are exactly the same as for the builtin :func:`open`.

    :param limiter: an optional capacity limiter to use with the file
        instead of the default one
    :return: an asynchronous file object

    .. versionchanged:: 4.14.0
        Added the ``limiter`` keyword argument.

    """
    fp = await to_thread.run_sync(
        open,
        file,
        mode,
        buffering,
        encoding,
        errors,
        newline,
        closefd,
        opener,
        limiter=limiter,
    )
    return AsyncFile(fp, limiter=limiter)


def wrap_file(
    file: IO[AnyStr], *, limiter: CapacityLimiter | None = None
) -> AsyncFile[AnyStr]:
    """
    Wrap an existing file as an asynchronous file.

    :param file: an existing file-like object
    :param limiter: an optional capacity limiter to use with the file
        instead of the default one
    :return: an asynchronous file object

    .. versionchanged:: 4.14.0
        Added the ``limiter`` keyword argument.

    """
    return AsyncFile(file, limiter=limiter)


@dataclass(eq=False)
class _PathIterator(AsyncIterator[T]):
    iterator: Iterator[PathLike[str]]
    limiter: CapacityLimiter | None
    # This was added to ensure that iterating over a subclass of Path yields instances
    # of that subclass rather than the base Path class.
    path_cls: type[T]

    async def __anext__(self) -> T:
        nextval = await to_thread.run_sync(
            next, self.iterator, None, abandon_on_cancel=True, limiter=self.limiter
        )
        if nextval is None:
            raise StopAsyncIteration from None

        return self.path_cls(nextval, limiter=self.limiter)


class Path:
    """
    An asynchronous version of :class:`pathlib.Path`.

    This class cannot be substituted for :class:`pathlib.Path` or
    :class:`pathlib.PurePath`, but it is compatible with the :class:`os.PathLike`
    interface.

    It implements the Python 3.10 version of :class:`pathlib.Path` interface, except for
    the deprecated :meth:`~pathlib.Path.link_to` method.

    Some methods may be unavailable or have limited functionality, based on the Python
    version:

    * :meth:`~pathlib.Path.copy` (available on Python 3.14 or later)
    * :meth:`~pathlib.Path.copy_into` (available on Python 3.14 or later)
    * :meth:`~pathlib.Path.from_uri` (available on Python 3.13 or later)
    * :meth:`~pathlib.PurePath.full_match` (available on Python 3.13 or later)
    * :attr:`~pathlib.Path.info` (available on Python 3.14 or later)
    * :meth:`~pathlib.Path.is_junction` (available on Python 3.12 or later)
    * :meth:`~pathlib.PurePath.match` (the ``case_sensitive`` parameter is only
      available on Python 3.13 or later)
    * :meth:`~pathlib.Path.move` (available on Python 3.14 or later)
    * :meth:`~pathlib.Path.move_into` (available on Python 3.14 or later)
    * :meth:`~pathlib.PurePath.relative_to` (the ``walk_up`` parameter is only available
      on Python 3.12 or later)
    * :meth:`~pathlib.Path.walk` (available on Python 3.12 or later)

    Any methods that do disk I/O need to be awaited on. These methods are:

    * :meth:`~pathlib.Path.absolute`
    * :meth:`~pathlib.Path.chmod`
    * :meth:`~pathlib.Path.cwd`
    * :meth:`~pathlib.Path.exists`
    * :meth:`~pathlib.Path.expanduser`
    * :meth:`~pathlib.Path.group`
    * :meth:`~pathlib.Path.hardlink_to`
    * :meth:`~pathlib.Path.home`
    * :meth:`~pathlib.Path.is_block_device`
    * :meth:`~pathlib.Path.is_char_device`
    * :meth:`~pathlib.Path.is_dir`
    * :meth:`~pathlib.Path.is_fifo`
    * :meth:`~pathlib.Path.is_file`
    * :meth:`~pathlib.Path.is_junction`
    * :meth:`~pathlib.Path.is_mount`
    * :meth:`~pathlib.Path.is_socket`
    * :meth:`~pathlib.Path.is_symlink`
    * :meth:`~pathlib.Path.lchmod`
    * :meth:`~pathlib.Path.lstat`
    * :meth:`~pathlib.Path.mkdir`
    * :meth:`~pathlib.Path.open`
    * :meth:`~pathlib.Path.owner`
    * :meth:`~pathlib.Path.read_bytes`
    * :meth:`~pathlib.Path.read_text`
    * :meth:`~pathlib.Path.readlink`
    * :meth:`~pathlib.Path.rename`
    * :meth:`~pathlib.Path.replace`
    * :meth:`~pathlib.Path.resolve`
    * :meth:`~pathlib.Path.rmdir`
    * :meth:`~pathlib.Path.samefile`
    * :meth:`~pathlib.Path.stat`
    * :meth:`~pathlib.Path.symlink_to`
    * :meth:`~pathlib.Path.touch`
    * :meth:`~pathlib.Path.unlink`
    * :meth:`~pathlib.Path.walk`
    * :meth:`~pathlib.Path.write_bytes`
    * :meth:`~pathlib.Path.write_text`

    Additionally, the following methods return an async iterator yielding
    :class:`~.Path` objects:

    * :meth:`~pathlib.Path.glob`
    * :meth:`~pathlib.Path.iterdir`
    * :meth:`~pathlib.Path.rglob`

    .. versionchanged:: 4.14.0
        Added the ``limiter`` keyword argument.
    """

    __slots__ = "__weakref__", "_limiter", "_path"

    __weakref__: Any

    def __init__(
        self, *args: str | PathLike[str], limiter: CapacityLimiter | None = None
    ) -> None:
        if limiter is not None and not isinstance(limiter, CapacityLimiter):
            raise TypeError(
                f"limiter must be a CapacityLimiter or None, not "
                f"{limiter.__class__.__name__}"
            )

        self._path: Final[pathlib.Path] = pathlib.Path(*args)
        self._limiter = limiter

    def __fspath__(self) -> str:
        return self._path.__fspath__()

    if sys.version_info >= (3, 15):

        def __vfspath__(self) -> str:
            return self._path.__vfspath__()

    def __str__(self) -> str:
        return self._path.__str__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.as_posix()!r})"

    def __bytes__(self) -> bytes:
        return self._path.__bytes__()

    def __hash__(self) -> int:
        return self._path.__hash__()

    def __eq__(self, other: object) -> bool:
        target = other._path if isinstance(other, Path) else other
        return self._path.__eq__(target)

    def __lt__(self, other: pathlib.PurePath | Path) -> bool:
        target = other._path if isinstance(other, Path) else other
        return self._path.__lt__(target)

    def __le__(self, other: pathlib.PurePath | Path) -> bool:
