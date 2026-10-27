        task: asyncio.Task[object],
        mode: Literal["read", "write"],
        lock_file: str,
        *,
        waiting_writer: bool,
    ) -> Literal["held", "enter"] | ConcurrentFuture[None]:
        with self._lock:
            if (depth := self._depths.get(task)) is not None:
                if self._mode != mode:
                    msg = (
                        f"Cannot acquire {mode} lock on {lock_file}: already holding a "
                        f"{'write' if mode == 'read' else 'read'} lock "
                        f"({'downgrade' if mode == 'read' else 'upgrade'} not allowed)"
                    )
                    raise RuntimeError(msg)
                self._depths[task] = depth + 1
                return "held"
            writers_ahead = self._waiting_writers - waiting_writer
            if not self._transitioning:
                if mode == "read" and self._mode == "read" and not writers_ahead:
                    self._depths[task] = 1
                    return "held"
                if not self._depths and (mode == "write" or not writers_ahead):
                    self._transitioning, self._mode = True, mode
                    return "enter"
            return self._changed

    async def release(
        self, *, force: bool, lock_file: str, leave: Callable[[], asyncio.Future[_BackendOutcome[None]]]
    ) -> None:
        task = _current_task()
        with self._lock:
            if (depth := self._depths.get(task)) is None:
                if force:
                    return
                msg = f"Cannot release a lock on {lock_file} that is not held by this task"
                raise RuntimeError(msg)
            if not force and depth > 1:
                self._depths[task] = depth - 1
                return
            del self._depths[task]
            if self._depths:
                return
            self._transitioning = True
            mode = self._mode
        try:
            cancellation = await _drain_cancellation(leave())
        except BaseException:
            # The sync lock may still hold the transaction, so keep the hold for a retried release to find.
            self._finish_transition(mode=mode, holder=task)
            raise
        self._finish_transition(mode=None)
        if cancellation is not None:
            # The release succeeded, so a restored hold would be a phantom no backend call can clear.
            raise cancellation

    def reset(self) -> None:
        with self._lock:
            self._depths.clear()
            self._mode, self._transitioning = None, False
            self._notify()

    def _finish_transition(
        self, *, mode: Literal["read", "write"] | None, holder: asyncio.Task[object] | None = None
    ) -> None:
        with self._lock:
            self._transitioning, self._mode = False, mode
            if holder is not None:
                self._depths[holder] = 1
            self._notify()

    def _notify(self) -> None:
        changed, self._changed = self._changed, ConcurrentFuture()
        changed.set_result(None)


_TASK_OWNERS: Final[WeakKeyDictionary[ReadWriteLock | SoftReadWriteLock, _TaskOwners]] = WeakKeyDictionary()
_TASK_OWNERS_LOCK: Final[Lock] = Lock()


def _task_owners_for(sync_lock: ReadWriteLock | SoftReadWriteLock) -> _TaskOwners:
    """Singleton wrappers share one sync lock, so they must share its task holds too."""
    with _fork_transition(), _TASK_OWNERS_LOCK:
        if (owners := _TASK_OWNERS.get(sync_lock)) is None:
            owners = _TASK_OWNERS[sync_lock] = _TaskOwners()
        return owners


def _current_task() -> asyncio.Task[object]:
    if (task := asyncio.current_task()) is None:
        msg = "an async reader/writer lock must be used from inside an asyncio task"
        raise RuntimeError(msg)
    return task


async def _drain_cancellation(future: asyncio.Future[_BackendOutcome[None]]) -> asyncio.CancelledError | None:
    # Returned, not raised, so the caller can tell a call that completed under cancellation from one that failed.
    try:
        await _wait_until_done(future)
    except asyncio.CancelledError as cancellation:
        try:
            await _drain_future(future)
        except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
            _raise_cancelled_error(cancellation, error)
        return cancellation
    _future_result(future)
    return None


async def _drain_future(future: asyncio.Future[_BackendOutcome[_T]]) -> _T:
    while not future.done():
        with contextlib.suppress(asyncio.CancelledError):
            await _wait_until_done(future)
    return _future_result(future)


async def _wait_until_done(future: asyncio.Future[_T]) -> None:
    if not future.done():
        await asyncio.wait((future,))


def _future_result(future: asyncio.Future[_BackendOutcome[_T]]) -> _T:
    outcome = future.result()
    if (error := outcome.error) is None:
        return cast("_T", outcome.value)
    context = error.__context__
    try:
        raise error  # ruff:ignore[raise-within-try]  # the handler restores context changed across the async boundary
    except BaseException:
        error.__context__ = context
        raise


def _run_in_executor(
    loop: asyncio.AbstractEventLoop, executor: Executor | None, func: Callable[[], _T]
) -> asyncio.Future[_BackendOutcome[_T]]:
    outcome: asyncio.Future[_BackendOutcome[_T]] = loop.create_future()

    def settle(call: asyncio.Future[_BackendOutcome[_T]]) -> None:
        try:
            result = call.result()
        except asyncio.CancelledError:
            # shutdown(cancel_futures=True) dropped the call unrun: a CancelledError would cancel a task nobody canceled
            # and hide that the lock state never changed.
            msg = "the executor shut down before running the lock backend call"
            result = _BackendOutcome(error=RuntimeError(msg))
        outcome.set_result(result)

    loop.run_in_executor(executor, _capture_call, func).add_done_callback(settle)
    return outcome


def _capture_call(func: Callable[[], _T]) -> _BackendOutcome[_T]:
    try:
        return _BackendOutcome(value=func())
    except BaseException as error:  # ruff:ignore[blind-except]  # backend control-flow exceptions are operation results
        return _BackendOutcome(error=error)


def _raise_cancelled_error(cancellation: asyncio.CancelledError, error: BaseException) -> NoReturn:
    # A reconciliation step failed while unwinding a cancellation, so keep both exception chains. Splice the error's
    # existing context onto the cancellation, then make the cancellation the error's context, so both the failure and
    # the cancellation that triggered it survive. Shared by the async wrappers so cancellations report the same way.
    if (context := error.__context__) is not None and context is not cancellation:
        if (cancellation_context := cancellation.__context__) is not None:
            _append_exception_context(context, cancellation_context)
        cancellation.__context__ = context
    error.__context__ = cancellation
    _raise_chained_errors(error)


async def _capture_awaitable(awaitable: Awaitable[_T]) -> _BackendOutcome[_T]:
    try:
        return _BackendOutcome(value=await awaitable)
    except BaseException as error:  # ruff:ignore[blind-except]  # backend cancellation must remain distinct from caller cancellation
        return _BackendOutcome(error=error)


__all__ = [
    "_AsyncTransitionGate",
    "_AsyncTransitionUnavailableError",
    "_BackendOutcome",
    "_capture_awaitable",
    "_capture_call",
    "_drain_future",
    "_future_result",
    "_raise_cancelled_error",
    "_run_in_executor",
    "_task_owners_for",
    "_wait_until_done",
]
<|endoftext|>"""
The generation-log protocol behind :class:`~filelock.SoftReadWriteLock`.

The lock's whole state is a sequence of immutable snapshots, ``gen/<N>``, each naming the writer and the readers that
hold the lock at generation ``N``. Every transition is one atomic publication of the next snapshot: a participant reads
the latest generation, derives the successor it wants, and links a fully written temporary file to ``gen/<N+1>``. The
link fails if a peer got there first, so the participant re-reads and tries again. No step is ever half done: a crash
leaves at most an unlinked temporary file, never a state that another participant must repair before it can proceed.

Liveness rests on a nonce, not a clock. Each participant keeps ``holders/<token>`` and rewrites its nonce every
heartbeat. A contender records the bytes it read there and the moment it read them on its own monotonic clock; a
member whose bytes have not changed for ``stale_threshold`` is evicted by committing a snapshot without it. Two hosts
never compare clocks, so a skewed file server or a suspended virtual machine cannot make a live holder look dead.

The generation number is monotonic and unique to one transition within one incarnation of the log, so the generation at
which a hold was granted fences it: a resource that rejects anything carrying a lower generation than the highest it has
accepted refuses a holder that paused past its expiry and resumed. A random epoch file names the incarnation, so a
participant that outlives a removal of the log reads the new one instead of continuing the old numbering into it.
"""

from __future__ import annotations

import os
import secrets
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from filelock._error import SoftFileLockProtocolError
from filelock._identity import host_name

if TYPE_CHECKING:
    from collections.abc import Callable

Mode = Literal["read", "write"]
HeartbeatOutcome = Literal["ok", "lost", "transient"]

_PROTOCOL: Final[str] = "filelock-rw/1"
_GENERATION_DIGITS: Final[int] = 20
_TOKEN_HEX_LENGTH: Final[int] = 32
#: A snapshot line is ``reader=`` plus a token, so this bounds a hostile record without capping real reader counts.
_MAX_RECORD_SIZE: Final[int] = 1 << 20
#: Snapshots older than this many generations behind the latest are removed by whoever commits. The window is also how
#: far ``latest`` probes past a gap before it trusts that nothing newer exists; ``gen/HEAD``, not the window, is what
#: keeps a listing served further behind from passing an old generation off as the head.
_RETAINED_GENERATIONS: Final[int] = 64
_GENERATIONS_DIRECTORY: Final[str] = "gen"
#: A random token created once per log: removing ``gen/`` removes it, so a participant can tell a log that was wiped and
#: restarted from the one it remembers.
_EPOCH_NAME: Final[str] = "epoch"
#: The epoch plus a copy of the newest snapshot, renamed into place after each commit. NFS revalidates a file opened by
#: name but may serve a listing from cache, so this is the head a stale client cannot miss; it may lag, never lead.
_HEAD_NAME: Final[str] = "HEAD"
_HOLDERS_DIRECTORY: Final[str] = "holders"
_TEMPORARY_PREFIX: Final[str] = ".commit-"


class Files(Protocol):
    """The filesystem operations the protocol runs on; :mod:`._storage` implements them over :mod:`os`."""

    def read(self, path: str) -> bytes | None:
        """The file's bytes, ``None`` when it does not exist, or ``b""`` when it is not a regular file."""

    def create(self, path: str, data: bytes, *, durable: bool = False) -> None:
        """
        Create the file exclusively with the whole record written; raise ``FileExistsError`` when it exists.

        *durable* flushes it to stable storage first, for a record that outlives the host that wrote it: a generation a
        crash left empty would wedge every other participant. A holder record dies with its writer, so it skips that.
        """

    def link(self, source: str, target: str) -> bool:
        """Hard-link *source* to *target* without replacing; ``True`` when *target* names *source*'s file afterwards."""

    def replace(self, source: str, target: str) -> bool:
        """Rename *source* over *target* atomically; ``False`` when the platform refused because *target* is in use."""

    def overwrite(self, path: str, data: bytes) -> bool:
        """Rewrite the file in place from offset zero; ``False`` when it no longer exists."""

    def unlink(self, path: str) -> None:
        """Remove the file, tolerating one that is already gone."""

    def listdir(self, path: str) -> list[str]:
        """The names in the directory, or an empty list when it does not exist."""

    def prepare(self, root: str, *, force: bool = False) -> None:
        """
        Create the protocol directories under *root*, refusing anything at those paths that is not a directory.

        Done once per lock; *force* repeats it after a write found a directory missing.
        """


@dataclass(frozen=True)
class Snapshot:
    """The lock's state at one generation: which token writes, which tokens read."""

    generation: int
    writer: str | None
    readers: frozenset[str]

    @property
    def members(self) -> frozenset[str]:
        """Every token this generation names."""
        return self.readers if self.writer is None else self.readers | {self.writer}

    def encode(self) -> bytes:
        lines = [_PROTOCOL, f"generation={self.generation}"]
        if self.writer is not None:
            lines.append(f"writer={self.writer}")
        lines.extend(f"reader={token}" for token in sorted(self.readers))
        return "".join(f"{line}\n" for line in lines).encode("ascii")


def parse_snapshot(data: bytes) -> Snapshot | None:
    """The snapshot a ``gen/<N>`` record encodes, or ``None`` when the record is malformed."""
    if not data or len(data) > _MAX_RECORD_SIZE:
        return None
    try:
        lines = data.decode("ascii").split("\n")
    except UnicodeDecodeError:
        return None
    if lines[0] != _PROTOCOL or lines[-1]:
        return None
    generation: int | None = None
    writer: str | None = None
    readers: set[str] = set()
    for line in lines[1:-1]:
        key, _, value = line.partition("=")
        if (
            key == "generation"
            and generation is None
            and len(value) <= _GENERATION_DIGITS
            and value.isdigit()
            and str(int(value)) == value
        ):
            generation = int(value)
        elif key == "writer" and writer is None and is_token(value):
            writer = value
        elif key == "reader" and is_token(value) and value not in readers:
            readers.add(value)
        else:
            return None
    return None if generation is None else Snapshot(generation=generation, writer=writer, readers=frozenset(readers))


def is_token(value: str) -> bool:
    """Whether *value* is a claim token: exactly 32 lowercase hex digits."""
    return len(value) == _TOKEN_HEX_LENGTH and all(character in "0123456789abcdef" for character in value)


def new_token() -> str:
    """A fresh claim token."""
    return secrets.token_hex(_TOKEN_HEX_LENGTH // 2)


def encode_holder(token: str, nonce: str, lease: float) -> bytes:
    """
    The ``holders/<token>`` record; only the nonce changes between heartbeats, so the length stays constant.

    *lease* is the writer's own ``stale_threshold``: a peer configured tighter must not read this holder's longer
    heartbeat interval as death, so evictors wait the longer of their threshold and this one.
    """
    return (
        f"{_PROTOCOL}\ntoken={token}\npid={os.getpid()}\nhost={host_name()}\nlease={float(lease)!r}\nnonce={nonce}\n"
    ).encode("ascii")


def holder_lease(record: bytes | None) -> float | None:
    """The ``lease`` a holder record carries, or ``None`` when it has none (an older release) or it is malformed."""
    try:
        text = next(
            line for line in (record or b"").decode("ascii").split("\n") if line.startswith("lease=")
        ).removeprefix("lease=")
        lease = float(text)
    except (StopIteration, ValueError):  # UnicodeDecodeError is a ValueError
        return None
    # Only the exact spelling repr() writes: float() alone also takes "1_000", " 1", and non-ASCII digits.
    return lease if isfinite(lease) and lease > 0 and repr(lease) == text else None


class Ledger:
    """
    How long each observed value has stayed the same, measured on this observer's monotonic clock.

    A value that has not changed for ``stale_threshold`` belongs to a participant that stopped refreshing it. The clock
    is the observer's own and only ever measures the gap between two of its own observations, which is what lets two
    hosts agree on liveness without agreeing on the time.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._seen: dict[str, tuple[bytes | None, float]] = {}

    def observe(self, key: str, value: bytes | None) -> float:
        """Record *value* for *key* and return how long it has been unchanged."""
        now = self._clock()
        if (seen := self._seen.get(key)) is None or seen[0] != value:
            self._seen[key] = (value, now)
            return 0.0
        return now - seen[1]

    def forget(self, key: str) -> None:
        self._seen.pop(key, None)

    def retain(self, keys: set[str]) -> None:
        """Drop every key not in *keys*: peers that left would otherwise stay in the ledger for the lock's lifetime."""
        self._seen = {key: seen for key, seen in self._seen.items() if key in keys}


class GenerationLog:
    """The ordered snapshots under ``gen/``, found by listing and probing, advanced by linking the next one."""

    def __init__(self, files: Files, lock_file: str, root: str) -> None:
        self._files = files
        self._lock_file = lock_file
        self._directory = Path(root, _GENERATIONS_DIRECTORY)
        self._epoch_path = str(self._directory / _EPOCH_NAME)
        self._head_path = str(self._directory / _HEAD_NAME)
        self._epoch: bytes | None = None
        self._latest: Snapshot | None = None

    def latest(self) -> Snapshot:
        """
        The newest snapshot.

        Starts from the newest readable generation among those ``gen/HEAD``, the listing, or memory name, then probes
        forward by name across the retained window. ``HEAD`` only names a generation, and that generation must still
        read back: a committer stalled before its rename can leave ``HEAD`` on a generation compacted since. A
        generation compacted between the listing and the read, and a hole left by a committer that died before it
        compacted, are covered by the probe. Memory and ``HEAD`` are bound to the log's epoch, so a log wiped and
        restarted is never continued from a generation it no longer holds. Only a log that names generations, now or as
        remembered, of which none can be read is refused, because that is a client so far behind the log that nothing
        it reads can be trusted.
        """
        # When everything the first pass named is gone, peers compacted past it while this client looked, or the log was
        # replaced; one more pass settles whether this client cannot see the head at all.
        if (latest := self._find()) is None and (latest := self._find()) is None:
            raise SoftFileLockProtocolError(self._lock_file, None, "generation log names no readable snapshot")
        self._latest = latest
        return latest

    def _find(self) -> Snapshot | None:
        if (epoch := self._incarnation()) != self._epoch:
            self._epoch, self._latest = epoch, None
        named = set(self._list())
        # Generation 0 is the empty log, which has no record to read back.
        if self._latest is not None and self._latest.generation:
            named.add(self._latest.generation)
        if head := self._head_generation():
            named.add(head)
        for generation in sorted(named, reverse=True):
            if (start := self._read(generation)) is not None:
                # Compaction keeps the surviving generations contiguous, so the first missing name past a readable HEAD
                # generation is the end; probing the whole window on every poll cost 64 failed opens.
                return self._probe(start, patience=1 if generation == head else _RETAINED_GENERATIONS)
        newest = max(named, default=0)
        found = self._probe(Snapshot(generation=newest, writer=None, readers=frozenset()))
        return found if found.generation > newest or not named else None

    def _head_generation(self) -> int | None:
        # HEAD is never flushed, so a crash can leave it unreadable; readers verify what it names anyway, so a torn one
        # only sends them to the window probe.
        if (record := self._head_record()) is None or (snapshot := parse_snapshot(record)) is None:
            return None
        return snapshot.generation

    def _head_record(self) -> bytes | None:
        if (data := self._files.read(self._head_path)) is None:
            return None
        # A HEAD from another epoch landed after its log was removed; a log whose epoch was never written runs without.
        epoch, newline, record = data.partition(b"\n")
        return record if self._valid_epoch() == epoch + newline else None

    def _valid_epoch(self) -> bytes | None:
        epoch = self._epoch
        return epoch if epoch is not None and is_token(epoch.decode("ascii", "replace").removesuffix("\n")) else None

    def _incarnation(self) -> bytes | None:
        if (epoch := self._files.read(self._epoch_path)) is None:
            # Exclusive creation, not a link: reading the log must keep working where os.link does not exist. A peer
            # that reads the file before its token is written takes the empty value for the epoch, and its next commit
            # sees the change and re-reads, so a partial read costs one retry.
            try:
                self._files.create(self._epoch_path, f"{new_token()}\n".encode("ascii"), durable=True)
            except FileExistsError:
                pass
            except FileNotFoundError:
                # No gen/ yet: there is no log to tell apart, and the first participant's prepare() creates it.
                return None
            epoch = self._files.read(self._epoch_path)
        return epoch

    def _list(self) -> list[int]:
        # isdigit() alone admits Unicode digits: superscripts int() rejects, fullwidth ones it parses to another number.
        listed = sorted(
            int(name)
            for name in self._files.listdir(str(self._directory))
            if len(name) == _GENERATION_DIGITS and name.isascii() and name.isdigit()
        )
        for generation in listed[:-_RETAINED_GENERATIONS]:
            self._files.unlink(self._path(generation))
        return listed[-_RETAINED_GENERATIONS:]

    def _probe(self, start: Snapshot, patience: int = _RETAINED_GENERATIONS) -> Snapshot:
        latest = start
        generation = start.generation
        gap = 0
        while gap < patience:
            generation += 1
            if (snapshot := self._read(generation)) is None:
                gap += 1
                continue
            latest = snapshot
            gap = 0
        return latest

    def _read(self, generation: int) -> Snapshot | None:
        if (data := self._files.read(self._path(generation))) is None:
            return None
        if (snapshot := parse_snapshot(data)) is None or snapshot.generation != generation:
            name = self._name(generation)
            raise SoftFileLockProtocolError(self._lock_file, name, "malformed generation record")
        return snapshot

    def commit(self, successor: Snapshot) -> bool:
        """
        Publish *successor* as the next generation; ``False`` when a peer published that generation first.

        The record is complete before it gets its public name, and a no-replace hard link is the compare-and-swap: only
        one link to ``gen/<N+1>`` can ever succeed. The link's return code is not trusted, because an NFS retry can
        report a link that did land as failed; the file's identity after the call is what decides. A log whose epoch
        changed since the snapshot was read is a different log, so the commit is refused as lost and the caller
        re-reads; that narrows a wipe racing this commit to the moment between the epoch check and the link. A slot
        compaction freed is refused the same way, before the link and again after it, since a committer can stall in
        between. A record refused after it landed is removed: it is the oldest survivor by then, and a legitimate one
        was about to be compacted.
        """
        if self._files.read(self._epoch_path) != self._epoch or not self._extends_the_log(successor.generation):
            return False
        if not self._publish(
            self._path(successor.generation), data := successor.encode(), self._files.link, durable=True
        ):
            return False
        if not self._extends_the_log(successor.generation):
            self._files.unlink(self._path(successor.generation))
            return False
        self._latest = successor
        # Readers verify the generation HEAD names, so a failed rename costs a probe, not the commit. A late one would
        # move HEAD back past later commits, so HEAD only ever moves forward.
        with suppress(OSError):
            if (
                (epoch := self._valid_epoch()) is not None
                and self._files.read(self._epoch_path) == epoch
                and (self._head_generation() or 0) < successor.generation
            ):
                self._publish(self._head_path, epoch + data, self._files.replace, durable=False)
        if (expired := successor.generation - _RETAINED_GENERATIONS - 1) > 0:
            self._compact_through(expired)
        return True

    def _extends_the_log(self, generation: int) -> bool:
        # Compaction removes oldest first, so a surviving predecessor proves this slot was never freed. The first
        # generation has none; a HEAD of this epoch means the log already moved past it.
        if generation > 1:
            return self._files.read(self._path(generation - 1)) is not None
        return self._head_record() is None

    def _publish(self, target: str, data: bytes, place: Callable[[str, str], bool], *, durable: bool) -> bool:
        temporary = str(self._directory / f"{_TEMPORARY_PREFIX}{new_token()}")
        try:
            self._files.create(temporary, data, durable=durable)
        except FileNotFoundError:
            # No gen/: the log was removed, or nobody has prepared it yet. Either way there is nothing to publish into
            # until a participant's prepare() creates it, and the caller re-reads whatever log exists by then.
            return False
        try:
            return place(temporary, target)
        finally:
            self._files.unlink(temporary)

    def _compact_through(self, expired: int) -> None:
        # Oldest first, so survivors stay contiguous: a stalled committer's leftover would otherwise sit below a hole.
        oldest = expired
        while oldest > 1 and self._files.read(self._path(oldest - 1)) is not None:
            oldest -= 1
        for generation in range(oldest, expired + 1):
            self._files.unlink(self._path(generation))

    def _path(self, generation: int) -> str:
        return str(self._directory / self._name(generation))

    @staticmethod
    def _name(generation: int) -> str:
        return f"{generation:0{_GENERATION_DIGITS}d}"


class Participant:
    """
    One acquisition, from publishing a claim through holding the lock to leaving.

    ``advance`` is polled until it reports the lock granted, ``heartbeat`` runs on the holder's schedule, and ``leave``
    commits the participant out. None of them sleep: the caller owns the deadline and the poll interval. The ledger and
    the log outlive one acquisition: how long a peer's record has stayed unchanged is knowledge every attempt against
    the same lock shares, or a caller whose attempts are each shorter than the stale threshold could never evict.
    """

    def __init__(  # ruff:ignore[too-many-arguments]  # one value per protocol input; the lock builds this once per acquisition
        self,
        files: Files,
        root: str,
        mode: Mode,
        *,
        stale_threshold: float,
        clock: Callable[[], float],
        ledger: Ledger,
        log: GenerationLog,
    ) -> None:
        self.token: Final[str] = new_token()
        self.mode: Final[Mode] = mode
        self.generation: int | None = None
        self._files = files
        self._root = root
        self._stale_threshold = stale_threshold
        self._clock = clock
        self._ledger = ledger
        self._log = log
        self._holder = self._holder_path(self.token)
        self._entered = False
        self._next_sweep = clock()

    def publish(self) -> None:
        """Create the holder record peers watch for liveness; it exists before the token appears in any snapshot."""
        self._files.prepare(self._root)
        self._create_holder()

    def _create_holder(self) -> None:
        try:
            self._files.create(self._holder, self._record())
        except FileNotFoundError:
            # Removing the whole log takes the directories with it.
            self._files.prepare(self._root, force=True)
            self._files.create(self._holder, self._record())

    def _record(self) -> bytes:
        return encode_holder(self.token, new_token(), self._stale_threshold)

    def advance(self) -> bool:
        """
        Take the next step toward holding the lock; ``True`` once the lock is granted.

        A reader enters as soon as no live writer is named. A writer enters as soon as no live writer is named, which
        blocks every new reader, then waits for the named readers to leave. Stale members are evicted in the same commit
        that admits this participant, and a writer's final admission is itself a commit, so no admission can rest on a
        snapshot a peer is about to supersede.
        """
        # A lost race is not a reason to wait: re-read and try again before handing the poll interval back to the
        # caller.
        for _ in range(4):
            latest = self._log.latest()
            self._keep_claim_fresh(latest)
            self._sweep(latest)
            stale = frozenset(token for token in latest.members - {self.token} if self._is_stale(token))
            writer = None if latest.writer in stale else latest.writer
            readers = latest.readers - stale
            if not self._entered:
                if writer is not None:
                    # A live writer blocks entry in both modes; only an eviction is worth committing.
                    if stale:
                        self._commit(latest, writer, readers, stale)
                    return False
                writer, readers = (self.token, readers) if self.mode == "write" else (None, readers | {self.token})
            granted = self.mode == "read" or not readers
            if self._entered and not stale and not granted:
                return False
            if (generation := self._commit(latest, writer, readers, stale)) is None:
                continue
            self._entered = True
            if granted:
                self.generation = generation
            return granted
        return False

    def _keep_claim_fresh(self, latest: Snapshot) -> None:
        # The snapshot is the truth about membership: a commit the filesystem reported lost may have landed, and a peer
        # may have evicted this participant while it waited to drain, so both directions are taken from it. A wait can
        # outlast the stale threshold, so the nonce changes on every poll; a peer then never reads a patient contender
        # as a corpse. A record that has gone missing was taken by an evictor or a sweeper that did, or by a removal of
        # the whole log.
        self._entered = self.token in latest.members
        if not self._refresh_nonce():
            with suppress(FileExistsError):
                self._create_holder()

    def _sweep(self, latest: Snapshot) -> None:
        # A participant that crashed before its first commit leaves a record no snapshot names, and a commit that
        # crashed between creating and unlinking its temporary file leaves that too. Neither blocks anyone, so they are
        # collected on the same unchanged-for-a-threshold rule as a member, and only every half threshold at most.
        if (now := self._clock()) < self._next_sweep:
            return
        self._next_sweep = now + self._stale_threshold / 2
        watched = set(latest.members)
        holders = Path(self._root, _HOLDERS_DIRECTORY)
        for name in self._files.listdir(str(holders)):
            if name != self.token and name not in latest.members and is_token(name):
                watched.add(key := f"holder:{name}")
                self._collect(key, str(holders / name))
        generations = Path(self._root, _GENERATIONS_DIRECTORY)
        for name in self._files.listdir(str(generations)):
            if name.startswith(_TEMPORARY_PREFIX):
                watched.add(key := f"temporary:{name}")
                self._collect(key, str(generations / name))
        self._ledger.retain(watched)

    def _collect(self, key: str, path: str) -> None:
        if self._unchanged_for_its_lease(key, path):
            self._files.unlink(path)
            self._ledger.forget(key)

    def _commit(
        self, latest: Snapshot, writer: str | None, readers: frozenset[str], stale: frozenset[str]
    ) -> int | None:
        successor = Snapshot(generation=latest.generation + 1, writer=writer, readers=readers)
        if not self._log.commit(successor):
            return None
        for token in stale:
            self._files.unlink(self._holder_path(token))
            self._ledger.forget(token)
        return successor.generation

    def _is_stale(self, token: str) -> bool:
        return self._unchanged_for_its_lease(token, self._holder_path(token))

    def _unchanged_for_its_lease(self, key: str, path: str) -> bool:
        record = self._files.read(path)
        return self._ledger.observe(key, record) >= max(self._stale_threshold, holder_lease(record) or 0.0)

    def _holder_path(self, token: str) -> str:
        return str(Path(self._root, _HOLDERS_DIRECTORY, token))

    def _refresh_nonce(self) -> bool:
        return self._files.overwrite(self._holder, self._record())

    def heartbeat(self) -> tuple[HeartbeatOutcome, OSError | None]:
        """
        Refresh the nonce and confirm the latest snapshot still names this participant.

        ``lost`` means a peer evicted this participant or removed its holder record: the lock is no longer held, and the
        holder must stop using what it protects. ``transient`` is a filesystem error refreshing the nonce, worth
        retrying on the next tick. A failure to read the log or to sweep decides nothing about this holder's liveness,
        so it is not counted against it.
        """
        try:
            if not self._refresh_nonce():
                return "lost", None
        except OSError as error:
            return "transient", error
        try:
            latest = self._log.latest()
            self._sweep(latest)
        except OSError:
            return "ok", None
        return ("ok" if self.token in latest.members else "lost"), None

    def leave(self) -> None:
        """Commit this participant out of the latest snapshot, then remove its holder record."""
        # The record goes even when the log cannot be read: a contender that never entered then leaves nothing behind,
        # and a member still named ages out the same way with or without one.
        try:
            while self.token in (latest := self._log.latest()).members:
                successor = Snapshot(
                    generation=latest.generation + 1,
                    writer=None if latest.writer == self.token else latest.writer,
                    readers=latest.readers - {self.token},
                )
                if self._log.commit(successor):
                    break
        finally:
            self._entered = False
            self.generation = None
            self._files.unlink(self._holder)


__all__ = [
    "Files",
    "GenerationLog",
    "HeartbeatOutcome",
    "Ledger",
    "Mode",
    "Participant",
    "Snapshot",
    "encode_holder",
    "holder_lease",
    "is_token",
    "new_token",
    "parse_snapshot",
]
<|endoftext|>"""Cross-process and cross-host reader/writer lock over a generation log of immutable snapshots."""

from __future__ import annotations

from ._async import AsyncAcquireSoftReadWriteReturnProxy, AsyncSoftReadWriteLock
from ._sync import SoftReadWriteLock

__all__ = [
    "AsyncAcquireSoftReadWriteReturnProxy",
    "AsyncSoftReadWriteLock",
    "SoftReadWriteLock",
]
<|endoftext|>"""Cross-process and cross-host reader/writer lock over a generation log of immutable snapshots."""

from __future__ import annotations

import atexit
import os
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Final
from weakref import WeakValueDictionary

from filelock._api import (
    AcquireReturnProxy,
    _canonical,
    _check_timeout_max,
    _ensure_current_process,
    _register_fork_class,
    _register_fork_object,
    _resolve_read_write_timeout,
    _seconds,
    _timeout_behavior,
)
from filelock._error import Timeout
from filelock._lease import LeaseCompromise

from ._protocol import GenerationLog, Ledger, Mode, Participant
from ._storage import OsFiles

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from filelock._lease import CompromiseReason

_ALL_INSTANCES: Final[WeakValueDictionary[int, SoftReadWriteLock]] = WeakValueDictionary()
_ALL_INSTANCES_LOCK: threading.Lock = threading.Lock()
_SINGLETONS_UNDER_CONSTRUCTION: Final[set[str]] = set()


class _SoftRWMeta(type):
    _instances: WeakValueDictionary[str, SoftReadWriteLock]
    _instances_lock: threading.RLock

    def __call__(  # ruff:ignore[too-many-arguments]  # forwards the public constructor's documented parameters
        cls,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,
        heartbeat_interval: float = 30.0,
        stale_threshold: float | None = None,
        poll_interval: float = 0.25,
        on_compromise: Callable[[LeaseCompromise], None] | None = None,
    ) -> SoftReadWriteLock:
        _ensure_current_process()
        timeout = _seconds("timeout", timeout)
        # Passed through only when set, so a subclass that declares its own constructor without it keeps working.
        extra = {} if on_compromise is None else {"on_compromise": on_compromise}
        if not is_singleton:
            return super().__call__(
                lock_file,
                timeout,
                blocking=blocking,
                is_singleton=is_singleton,
                heartbeat_interval=heartbeat_interval,
                stale_threshold=stale_threshold,
                poll_interval=poll_interval,
                **extra,
            )

        # Keyed like the log root, so one instance never spans two logs.
        normalized = _canonical(lock_file)
        with cls._instances_lock:
            instance = cls._instances.get(normalized)
            if instance is None:
                if normalized in _SINGLETONS_UNDER_CONSTRUCTION:  # pragma: needs fork
                    msg = f"Singleton lock construction is already active for {lock_file!s}"
                    raise RuntimeError(msg)
                construction_pid = os.getpid()
                _SINGLETONS_UNDER_CONSTRUCTION.add(normalized)
                try:
                    instance = super().__call__(
                        lock_file,
                        timeout,
                        blocking=blocking,
                        is_singleton=is_singleton,
                        heartbeat_interval=heartbeat_interval,
                        stale_threshold=stale_threshold,
                        poll_interval=poll_interval,
                        **extra,
                    )
                finally:
                    _SINGLETONS_UNDER_CONSTRUCTION.discard(normalized)
                if os.getpid() != construction_pid:  # pragma: needs fork
                    msg = "Lock construction cannot continue after fork; construct a new lock in the child"
                    raise RuntimeError(msg)
                cls._instances[normalized] = instance
            elif (
                _timeout_behavior(instance.timeout, any_negative=False)
                != _timeout_behavior(timeout, any_negative=False)
                or instance.blocking != blocking
            ):
                msg = (
                    f"Singleton lock created with timeout={instance.timeout}, blocking={instance.blocking},"
                    f" cannot be changed to timeout={timeout}, blocking={blocking}"
                )
                raise ValueError(msg)
            elif on_compromise is not None and instance._on_compromise != on_compromise:  # ruff: ignore[private-member-access]  # the metaclass owns the singleton it compares
                # A tuning difference keeps the first caller's values; a different loss callback is a different safety
                # contract, and returning the cached instance would silently drop it.
                msg = f"Singleton lock created with a different on_compromise callback for {lock_file!s}"
                raise ValueError(msg)
            else:
                _validate_intervals(heartbeat_interval, stale_threshold, poll_interval)
            return instance


class SoftReadWriteLock(metaclass=_SoftRWMeta):
    """
    Cross-process and cross-host reader/writer lock for shared filesystems.

    Use this class instead of :class:`~filelock.ReadWriteLock` when the lock file lives on a network filesystem (NFS,
    Lustre, HPC cluster shared storage). ``ReadWriteLock`` is backed by SQLite and cannot run on NFS because SQLite's
    ``fcntl`` locking is unreliable there.

    The lock's state is a log of immutable snapshots under ``foo.lock.rw/gen/<N>``, each naming the writer and the
    readers holding the lock at generation ``N``. Every acquire, release, and eviction publishes the next snapshot with
    one atomic no-replace hard link, so no transition can be left half done by a crash on any host. Each participant
    also keeps ``foo.lock.rw/holders/<token>``, a record whose nonce a daemon heartbeat thread rewrites every
    ``heartbeat_interval`` seconds. A contender evicts a member whose record has not changed for ``stale_threshold``
    seconds of the contender's own monotonic clock; no clock is ever compared across hosts, so skew between hosts or
    against the file server cannot make a live holder look dead.

    Writer acquire is writer-preferring: a writer enters the snapshot as soon as no live writer is named, which blocks
    any new reader, then waits for the named readers to leave. Writer starvation is impossible.

    An expired holder is not fenced by the lock: a process paused past ``stale_threshold`` resumes believing it holds
    the lock until its next heartbeat reports the loss through ``on_compromise``. :attr:`generation` is a monotonic
    fencing token for the protected resource: a store that rejects writes carrying a lower generation than the highest
    it has accepted refuses such a holder.

    Reentrancy, upgrade/downgrade rules, thread pinning, and singleton caching by canonical path match
    :class:`~filelock.ReadWriteLock`.

    Forking invalidates the inherited instance in the child so the child cannot double-own the lock with its parent;
    ``release()`` on that instance is a no-op, and the child must construct a new instance if it needs a lock.

    Trust boundary: protects against same-UID non-cooperating processes (one host or cross-host) and same-host
    different-UID users via ``0o600`` / ``0o700`` permissions. Does not protect against root compromise or multi-tenant
    mounts where hostile co-tenants share the UID.

    :param lock_file: path to the lock file; the protocol directory lives next to it as ``<lock_file>.rw``
    :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
    :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately on contention
    :param is_singleton: if ``True``, reuse existing instances for the same canonical path
    :param heartbeat_interval: seconds between heartbeat refreshes; default 30 s
    :param stale_threshold: seconds a holder record may stay unchanged before a contender evicts it; defaults to
        ``3 * heartbeat_interval``, matching etcd's ``LeaseKeepAlive`` convention
    :param poll_interval: seconds between acquire retries under contention; default 0.25 s
    :param on_compromise: called from the heartbeat thread with a :class:`~filelock.LeaseCompromise` when the hold is
        lost: a peer evicted it, or refreshes failed for long enough that a peer could have

    .. versionadded:: 3.27.0

    .. versionchanged:: 3.33.0

        The on-disk protocol is a generation log; earlier releases cannot share a lock path with this one.

    """

    _instances: WeakValueDictionary[str, SoftReadWriteLock] = WeakValueDictionary()
    _instances_lock = threading.RLock()

    def __init__(  # ruff:ignore[too-many-arguments]  # public constructor: one parameter per documented lock option
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,  # ruff:ignore[unused-method-argument]  # consumed by _SoftRWMeta.__call__
        heartbeat_interval: float = 30.0,
        stale_threshold: float | None = None,
        poll_interval: float = 0.25,
        on_compromise: Callable[[LeaseCompromise], None] | None = None,
    ) -> None:
        self._creator_pid = os.getpid()
        heartbeat_interval, stale_threshold, poll_interval = _validate_intervals(
            heartbeat_interval, stale_threshold, poll_interval
        )

        self.lock_file: str = os.fspath(lock_file)
        self.timeout: float = timeout
        self.blocking: bool = blocking
        self.heartbeat_interval: float = heartbeat_interval
        self.stale_threshold: float = stale_threshold
        self.poll_interval: float = poll_interval

        # Resolved once: a relative lock path must keep naming the same log after the process changes directory. The
        # final component stays unresolved, as in 4.0.12, so every release shares one log on a symlinked lock path.
        self._root = f"{_canonical(self.lock_file)}.rw"
        self._files = OsFiles(self.lock_file)
        self._ledger = Ledger(time.monotonic)
        self._log = GenerationLog(self._files, self.lock_file, self._root)
        self._on_compromise = on_compromise
        self._locks = _Locks(internal=threading.Lock(), transaction=threading.Lock())
        self._hold: _Hold | None = None
        self._compromise: LeaseCompromise | None = None
        self._closed: bool = False

        with _ALL_INSTANCES_LOCK:
            _ALL_INSTANCES[id(self)] = self
        _register_fork_object(self)

    @classmethod
    def _reset_class_after_fork(cls) -> None:  # pragma: forked child
        global _ALL_INSTANCES_LOCK  # ruff:ignore[global-statement]  # rebinds the module lock to a fresh one in the fork child
        _ALL_INSTANCES_LOCK = threading.Lock()
        cls._instances = WeakValueDictionary()
        cls._instances_lock = threading.RLock()
        _SINGLETONS_UNDER_CONSTRUCTION.clear()

    @property
    def generation(self) -> int | None:
        """
        The generation at which the current hold was granted, or ``None`` when no lock is held.

        Generations are monotonic across all participants and hosts, so this is a fencing token: pass it to the
        protected resource and have the resource reject any operation carrying a lower generation than the highest it
        has accepted. That refuses a holder that paused past ``stale_threshold``, was evicted, and resumed. Removing the
        ``<lock_file>.rw`` directory starts a new log whose generations begin again at 1.

        .. versionadded:: 3.33.0

        """
        with self._locks.internal:
            return None if self._hold is None else self._hold.participant.generation

    @property
    def compromise(self) -> LeaseCompromise | None:
        """
        How the current hold was lost, or ``None`` while it stands.

        Set by the heartbeat thread once a peer has evicted this holder or refreshes have failed for long enough that a
        peer could have. The holder should stop using the protected resource when it is set.

        .. versionadded:: 3.33.0

        """
        with self._locks.internal:
            return self._compromise

    @contextmanager
    def read_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> Generator[None]:
        """
        Context manager that acquires and releases a shared read lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :raises RuntimeError: if a write lock is already held on this instance
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self.acquire_read(timeout, blocking=blocking)
        try:
            yield
        finally:
            self.release()

    @contextmanager
    def write_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> Generator[None]:
        """
        Context manager that acquires and releases an exclusive write lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :raises RuntimeError: if a read lock is already held, or a write lock is held by a different thread
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self.acquire_write(timeout, blocking=blocking)
        try:
            yield
        finally:
            self.release()

    def acquire_read(self, timeout: float | None = None, *, blocking: bool | None = None) -> AcquireReturnProxy:
        """
        Acquire a shared read lock.

        If this instance already holds a read lock, the lock level is incremented (reentrant). Attempting to acquire a
        read lock while holding a write lock raises :class:`RuntimeError` (downgrade not allowed). On the 0→1
        transition the reader publishes its holder record, enters the next snapshot once no live writer is named, and
        starts a daemon heartbeat thread that rewrites the record's nonce every ``heartbeat_interval`` seconds so peers
        on other hosts do not evict it.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default; ``-1`` means block
            indefinitely
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable;
            ``None`` uses the instance default

        :returns: a proxy that can be used as a context manager to release the lock

        :raises RuntimeError: if a write lock is already held on this instance, if this instance was invalidated by
            :func:`os.fork`, or if :meth:`close` was called
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``
        :raises SoftFileLockProtocolError: if a snapshot cannot be read without risking overlap, or the filesystem
            refuses the no-replace hard links the protocol commits with

        """
        return self._acquire("read", timeout, blocking=blocking)

    def acquire_write(self, timeout: float | None = None, *, blocking: bool | None = None) -> AcquireReturnProxy:
        """
        Acquire an exclusive write lock.

        If this instance already holds a write lock from the same thread, the lock level is incremented (reentrant).
        Attempting to acquire a write lock while holding a read lock raises :class:`RuntimeError` (upgrade not
        allowed). Write locks are pinned to the acquiring thread: a different thread trying to re-enter also raises
        :class:`RuntimeError`.

        A writer enters the next snapshot as its writer as soon as no live writer is named, which blocks every new
        reader on every host, then waits for the readers that snapshot names to leave. Writer starvation is impossible:
        new readers see the named writer and wait behind it.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default; ``-1`` means block
            indefinitely
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable;
            ``None`` uses the instance default

        :returns: a proxy that can be used as a context manager to release the lock

        :raises RuntimeError: if a read lock is already held, if a write lock is held by a different thread, if this
            instance was invalidated by :func:`os.fork`, or if :meth:`close` was called
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``
        :raises SoftFileLockProtocolError: if a snapshot cannot be read without risking overlap, or the filesystem
            refuses the no-replace hard links the protocol commits with

        """
        return self._acquire("write", timeout, blocking=blocking)

    @classmethod
    def get_lock(
        cls,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
    ) -> SoftReadWriteLock:
        """
        Return the singleton :class:`SoftReadWriteLock` for *lock_file*.

        :param lock_file: path to the lock file; the protocol directory lives next to it as ``<lock_file>.rw``
        :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable

        :returns: the singleton lock instance

        :raises ValueError: if an instance already exists for this path with different *timeout* or *blocking* values

        """
        return cls(lock_file, timeout, blocking=blocking)

    def close(self) -> None:
        """
        Release any held lock and mark the instance closed.

        Idempotent. After calling this method the instance can no longer acquire locks — subsequent acquires raise
        :class:`RuntimeError`, and so does an acquire another thread is still waiting in, which gives up its claim
        instead of taking the lock. A fork-invalidated instance is closed without raising.
        """
        if self._creator_pid != os.getpid():  # pragma: forked child
            return
        # Closed first, under the lock an acquire publishes its hold under: an acquire still waiting then sees the flag
        # and leaves instead of publishing, and one that published before it is released below.
        with self._locks.internal:
            self._closed = True
        self.release(force=True)

    def release(self, *, force: bool = False) -> None:
        """
        Release one level of the current lock.

        When the lock level reaches zero the heartbeat thread is stopped, the participant is committed out of the next
        snapshot, and its holder record is removed. On a fork-invalidated instance (that is, the child of a
        :func:`os.fork` call made while the parent held a lock) this method is a no-op so inherited ``with`` blocks can
        unwind cleanly in the child.

        :param force: if ``True``, release the lock completely regardless of the current lock level

        :raises RuntimeError: if no lock is currently held and *force* is ``False``, or if a write lock is held by
            another thread and *force* is ``False``

        """
        if self._creator_pid != os.getpid():  # pragma: forked child
            return
        with self._locks.internal:
            hold = self._hold
            if hold is None:
                if force:
                    return
                msg = f"Cannot release a lock on {self.lock_file} (lock id: {id(self)}) that is not held"
                raise RuntimeError(msg)
            # Another thread's release would leave the generation log under the writer and let a peer in. The hold's
            # own heartbeat thread may release it, since on_compromise runs there.
            if (
                not force
                and hold.mode == "write"
                and (cur := threading.get_ident()) not in {hold.write_thread_id, hold.heartbeat_thread.ident}
            ):
                msg = (
                    f"Cannot release write lock on {self.lock_file} (lock id: {id(self)}) "
                    f"from thread {cur} while it is held by thread {hold.write_thread_id}"
                )
                raise RuntimeError(msg)
            if force:
                hold.level = 0
            else:
                hold.level -= 1
            if hold.level > 0:
                return
            self._hold = None

        # Stop the heartbeat before leaving, so a late tick cannot re-read a snapshot this participant just left and
        # report an eviction that was its own release. on_compromise runs on that thread and may be what called here,
        # in which case there is nothing to join.
        hold.heartbeat_stop.set()
        try:
            if hold.heartbeat_thread is not threading.current_thread():
                hold.heartbeat_thread.join(timeout=min(self.heartbeat_interval + 1.0, threading.TIMEOUT_MAX))
        finally:
            # The hold is already gone from this instance, so a join that raises must not strand its record too.
            hold.participant.leave()

    def _acquire(
        self,
        mode: Mode,
        timeout: float | None,
        *,
        blocking: bool | None,
    ) -> AcquireReturnProxy:
        if self._creator_pid != os.getpid():  # pragma: forked child
            msg = f"SoftReadWriteLock on {self.lock_file} was invalidated by fork(); construct a new instance"
            raise RuntimeError(msg)
        timeout = self.timeout if timeout is None else timeout
        blocking = self.blocking if blocking is None else blocking

        with self._locks.internal:
            if self._closed:
                raise self._closed_error()
            timeout = _resolve_read_write_timeout(timeout, blocking=blocking)
            if self._hold is not None:
                return self._validate_reentrant(mode)

        start = time.perf_counter()
        if not blocking:
            acquired = self._locks.transaction.acquire(blocking=False)
        elif timeout == -1:
            acquired = self._locks.transaction.acquire(blocking=True)
        else:
            # Lock.acquire raises OverflowError past TIMEOUT_MAX, which is 49.7 days on Windows.
            acquired = self._locks.transaction.acquire(blocking=True, timeout=min(timeout, threading.TIMEOUT_MAX))
        if not acquired:
            raise Timeout(self.lock_file) from None
        try:
            return self._do_acquire_inner(mode, timeout, start, blocking=blocking)
        finally:
            self._locks.transaction.release()

    def _closed_error(self) -> RuntimeError:
        return RuntimeError(f"SoftReadWriteLock on {self.lock_file} has been closed")

    def _do_acquire_inner(
        self,
        mode: Mode,
        effective_timeout: float,
        start: float,
        *,
        blocking: bool,
    ) -> AcquireReturnProxy:
        with self._locks.internal:
            if self._hold is not None:
                return self._validate_reentrant(mode)
        deadline = None if effective_timeout == -1 else start + effective_timeout
        participant = Participant(
            self._files,
            self._root,
            mode,
            stale_threshold=self.stale_threshold,
            clock=time.monotonic,
            ledger=self._ledger,
            log=self._log,
        )
        participant.publish()
        try:
            self._wait_for(participant.advance, deadline=deadline, blocking=blocking)
        except BaseException:
            # A contender that gives up must not stay named in the snapshot: a writer left there would block every
            # reader until a peer waits out the stale threshold. Its own token is all leave() touches, so it cannot
            # undo a peer's claim.
            with suppress(OSError):
                participant.leave()
            raise
        stop_event = threading.Event()
        hold = _Hold(
            level=1,
            mode=mode,
            write_thread_id=threading.get_ident() if mode == "write" else None,
            participant=participant,
            heartbeat_thread=_HeartbeatThread(
                refresh=lambda: self._refresh(participant, hold),
                interval=self.heartbeat_interval,
                stop_event=stop_event,
                name=f"filelock-heartbeat-{id(self):x}",
            ),
            heartbeat_stop=stop_event,
            last_refresh=time.monotonic(),
        )
        # Publish the hold and start its heartbeat under one internal-lock section, so a concurrent release() never
        # observes a hold whose thread has not started and joins it. If the OS refuses the thread, clear the hold and
        # leave: left in place, a peer evicts the unrefreshed record and acquires while this instance still believes it
        # holds the lock. A close() that landed while this acquire waited gets the same treatment: published, the hold
        # would keep its heartbeat and its slot on an instance nobody can release through.
        start_error: BaseException | None = None
        with self._locks.internal:
            if self._closed:
                start_error = self._closed_error()
            else:
                self._hold = hold
                self._compromise = None
                try:
                    hold.heartbeat_thread.start()
                except BaseException as error:  # ruff:ignore[blind-except]  # clear the slot below and re-raise
                    self._hold = None
                    start_error = error
        if start_error is not None:
            participant.leave()
            raise start_error
        return AcquireReturnProxy(lock=self)

    def _validate_reentrant(self, mode: Mode) -> AcquireReturnProxy:
        hold = self._hold
        assert hold is not None  # ruff:ignore[assert]  # callers dispatch here only inside the self._hold is not None branch
        if hold.mode != mode:
            opposite = "write" if mode == "read" else "read"
            direction = "downgrade" if mode == "read" else "upgrade"
            msg = (
                f"Cannot acquire {mode} lock on {self.lock_file} (lock id: {id(self)}): "
                f"already holding a {opposite} lock ({direction} not allowed)"
            )
            raise RuntimeError(msg)
        if mode == "write" and (cur := threading.get_ident()) != hold.write_thread_id:
            msg = (
                f"Cannot acquire write lock on {self.lock_file} (lock id: {id(self)}) "
                f"from thread {cur} while it is held by thread {hold.write_thread_id}"
            )
            raise RuntimeError(msg)
        hold.level += 1
        return AcquireReturnProxy(lock=self)

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        *,
        deadline: float | None,
        blocking: bool,
    ) -> None:
        while not predicate():
            # The flag is read without the lock here only to stop waiting early; publishing the hold re-checks it.
            if self._closed:
                raise self._closed_error()
            now = time.perf_counter()
            if not blocking:
                raise Timeout(self.lock_file)
            if deadline is not None and now >= deadline:
                raise Timeout(self.lock_file)
            sleep_for = self.poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(deadline - now, 0.0))
            time.sleep(sleep_for)

    def _refresh(self, participant: Participant, hold: _Hold) -> bool:
        # The loop ends at the first loss, so the holder hears about it once. A transient filesystem error (ESTALE /
        # EIO on the NFS-style filesystems this lock targets) is not a loss: retry rather than report a false
        # compromise. Report the record unrefreshable only once failures have run long enough that a peer could evict
        # it before the next success would land, a margin before the record actually ages out.
        outcome, error = participant.heartbeat()
        if outcome == "ok":
            hold.last_refresh = time.monotonic()
            return True
        if outcome == "lost":
            self._report_compromise(hold, "evicted", None)
            return False
        if time.monotonic() - hold.last_refresh >= self.stale_threshold - self.heartbeat_interval:
            self._report_compromise(hold, "refresh-failed", error)
            return False
        return True

    def _report_compromise(self, hold: _Hold, reason: CompromiseReason, error: OSError | None) -> None:
        # A tick that outlived its release's join must not stamp the old hold's loss onto whatever was acquired since.
        with self._locks.internal:
            if self._hold is not hold:
                return
            compromise = LeaseCompromise(
                lock_file=self.lock_file, token=hold.participant.token, reason=reason, error=error
            )
            self._compromise = compromise
        if self._on_compromise is not None:
            self._on_compromise(compromise)

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        self._locks = _Locks(internal=threading.Lock(), transaction=threading.Lock())
        self._hold = None


class _HeartbeatThread(threading.Thread):
    def __init__(
        self,
        refresh: Callable[[], bool],
        interval: float,
        stop_event: threading.Event,
        name: str,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._refresh = refresh
        self._interval = interval
        self._stop_event = stop_event

    def run(self) -> None:
        # Set on every exit, an on_compromise that raises included, so nothing waits on a thread that is gone.
        try:
            while not self._stop_event.wait(self._interval):
                if not self._refresh():
                    return
        finally:
            self._stop_event.set()


def _validate_intervals(
    heartbeat_interval: float, stale_threshold: float | None, poll_interval: float
) -> tuple[float, float, float]:
    heartbeat_interval = _seconds("heartbeat_interval", heartbeat_interval)
    if not isfinite(heartbeat_interval) or heartbeat_interval <= 0:
        msg = f"heartbeat_interval must be positive and finite, got {heartbeat_interval}"
        raise ValueError(msg)
    # The heartbeat thread waits this long between refreshes, and Event.wait overflows past TIMEOUT_MAX.
    _check_timeout_max("heartbeat_interval", heartbeat_interval)
    stale_threshold = (
        heartbeat_interval * 3 if stale_threshold is None else _seconds("stale_threshold", stale_threshold)
    )
    if not isfinite(stale_threshold) or stale_threshold <= heartbeat_interval:
        msg = (
            f"stale_threshold must exceed heartbeat_interval ({heartbeat_interval}) "
            f"and be finite, got {stale_threshold}"
        )
        raise ValueError(msg)
    if not isfinite(poll_interval := _seconds("poll_interval", poll_interval)) or poll_interval <= 0:
        msg = f"poll_interval must be positive and finite, got {poll_interval}"
        raise ValueError(msg)
    _check_timeout_max("poll_interval", poll_interval)
    # A waiting writer refreshes its record once per poll, so a poll slower than the threshold reads as dead.
    if poll_interval >= stale_threshold:
        msg = f"poll_interval must be below stale_threshold ({stale_threshold}), got {poll_interval}"
        raise ValueError(msg)
    return heartbeat_interval, stale_threshold, poll_interval


@dataclass
class _Locks:
    internal: threading.Lock
    transaction: threading.Lock


@dataclass
class _Hold:
    """Everything that exists only while a lock is held; ``None`` when the instance has no lock."""

    level: int
    mode: Mode
    write_thread_id: int | None
    participant: Participant
    heartbeat_thread: _HeartbeatThread
    heartbeat_stop: threading.Event
    last_refresh: float


def _cleanup_all_instances() -> None:  # pragma: no cover - runs from atexit at interpreter shutdown
    for instance in list(_ALL_INSTANCES.values()):
        with suppress(Exception):
            instance.release(force=True)


atexit.register(_cleanup_all_instances)
_register_fork_class(SoftReadWriteLock)


__all__ = [
    "SoftReadWriteLock",
]
<|endoftext|>"""The :class:`~._protocol.Files` implementation over :mod:`os`."""

from __future__ import annotations

import os
import stat
import sys
import time
from contextlib import suppress
from errno import ENOENT, ESTALE
from pathlib import Path
from typing import Final

from filelock._strict import _link_no_follow, _raise_if_hard_links_unsupported
from filelock._util import ensure_directory_exists, write_all

from ._protocol import _GENERATIONS_DIRECTORY, _HOLDERS_DIRECTORY, _MAX_RECORD_SIZE

_O_NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)
#: Windows opens descriptors in text mode by default, which rewrites newlines; every record here is exact bytes.
_O_BINARY: Final[int] = getattr(os, "O_BINARY", 0)
#: ESTALE is what an NFS client reports for a file a peer unlinked out from under its cached handle: gone, not broken.
_MISSING_ERRNOS: Final[frozenset[int]] = frozenset({ENOENT, ESTALE})
#: Windows refuses an open with EACCES while another process is deleting the file or holds it open without sharing;
#: both clear within moments, so an open is retried this long before the refusal is taken as real.
_WINDOWS_OPEN_GRACE: Final[float] = 0.5


class OsFiles:
    """Record files under a lock's protocol root, opened without following symlinks and owned by this UID alone."""

    def __init__(self, lock_file: str) -> None:
        self._lock_file = lock_file
        self._prepared = False

    @staticmethod
    def read(path: str) -> bytes | None:
        # O_NONBLOCK keeps an open of a FIFO planted at the path from blocking; the regular-file check below rejects it.
        if (fd := _open(path, os.O_RDONLY | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | _O_BINARY)) is None:
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return b""
            chunks: list[bytes] = []
            remaining = _MAX_RECORD_SIZE + 1
            while remaining > 0 and (chunk := os.read(fd, remaining)):
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    @classmethod
    def create(cls, path: str, data: bytes, *, durable: bool = False) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_BINARY, 0o600)
        try:
            write_all(fd, data)
            if durable:
                os.fsync(fd)
        except BaseException:
            os.close(fd)
            # The name is unique to this call, so removing it cannot take a record from anyone else.
            cls.unlink(path)
            raise
        os.close(fd)

    def link(self, source: str, target: str) -> bool:
        failure: NotImplementedError | OSError | None = None
        try:
            _link_no_follow(source, target)
        except FileExistsError:
            pass
        except (NotImplementedError, OSError) as error:
            _raise_if_hard_links_unsupported(self._lock_file, error)
            failure = error
        # Whatever the call reported, the file's identity decides: an NFS retransmit can turn a link that landed into
        # an error, and an EEXIST can be this very link answered twice. Any other error on a link that did not land is
        # a real fault, not a peer winning the race, and retrying it would spin.
        if (identity := _identity(source)) is not None and identity == _identity(target):
            return True
        if failure is not None:
            raise failure
        return False

    @staticmethod
    def replace(source: str, target: str) -> bool:
        try:
            Path(source).replace(target)
        except PermissionError:
            # Windows refuses to rename over a file a reader holds open; callers tolerate a target one rename behind.
            return False
        return True

    @staticmethod
    def overwrite(path: str, data: bytes) -> bool:
        if (fd := _open(path, os.O_WRONLY | _O_NOFOLLOW | _O_BINARY)) is None:
            return False
        try:
            write_all(fd, data)
        finally:
            os.close(fd)
        return True

    @staticmethod
    def unlink(path: str) -> None:
        try:
            Path(path).unlink()
        except OSError as error:
            # Windows refuses to remove a file another process holds open; the record then outlives its owner until
            # that process exits, which only delays the cleanup a later participant repeats.
            if error.errno not in _MISSING_ERRNOS and not isinstance(error, PermissionError):
                raise

    @staticmethod
    def listdir(path: str) -> list[str]:
        try:
            return [entry.name for entry in Path(path).iterdir()]
        except FileNotFoundError:
            return []

    def prepare(self, root: str, *, force: bool = False) -> None:
        if self._prepared and not force:
            return
        ensure_directory_exists(root)
        for directory in (Path(root), Path(root, _GENERATIONS_DIRECTORY), Path(root, _HOLDERS_DIRECTORY)):
            with suppress(FileExistsError):
                directory.mkdir(mode=0o700)
            # mkdir has no O_NOFOLLOW, so check what the name resolves to before anything is created inside it.
            mode = directory.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                msg = f"{directory} exists but is not a directory or is a symlink; refusing to use it"
                raise RuntimeError(msg)
        self._prepared = True


def _open(path: str, flags: int) -> int | None:
    # Missing is the one answer a caller acts on without raising. A refused open is real on POSIX, where nothing this
    # lock does makes its own records unreadable; on Windows it is usually another process mid-delete or holding the
    # file without sharing, so it is retried through a short grace before it counts, and a file still refusing after
    # that has been in delete-pending for longer than any deletion takes, so it is on its way out: missing.
    deadline: float | None = None
    while True:
        if isinstance(opened := _attempt_open(path, flags), int):
            return opened
        if opened.errno in _MISSING_ERRNOS:
            return None
        if not isinstance(opened, PermissionError) or sys.platform != "win32":
            raise opened
        if deadline is None:  # pragma: win32 cover
            deadline = time.monotonic() + _WINDOWS_OPEN_GRACE
        elif time.monotonic() >= deadline:  # pragma: win32 cover
            return None
        time.sleep(0.002)  # pragma: win32 cover


def _attempt_open(path: str, flags: int) -> int | OSError:
    try:
        return os.open(path, flags)
    except OSError as error:
        return error


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


__all__ = [
    "OsFiles",
]
<|endoftext|>"""Async wrapper around :class:`SoftReadWriteLock` for use with ``asyncio``."""

from __future__ import annotations

import asyncio
import functools
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar

from filelock._async import (
    _BackendOutcome,
    _capture_call,
    _drain_future,
    _future_result,
    _raise_cancelled_error,
    _task_owners_for,
    _wait_until_done,
)

from ._sync import SoftReadWriteLock

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from concurrent import futures
    from types import TracebackType

    from filelock._api import AcquireReturnProxy
    from filelock._lease import LeaseCompromise

_P = ParamSpec("_P")
_R = TypeVar("_R")


class AsyncSoftReadWriteLock:
    """
    Async wrapper around :class:`SoftReadWriteLock` for ``asyncio`` applications.

    The sync class's blocking filesystem operations run on a thread pool via ``loop.run_in_executor()``. Each
    ``asyncio`` task owns its own hold. Tasks share a read lock, and a task asking for the write lock waits until the
    other holders release, with waiting writers ahead of new readers. Upgrading or downgrading a held lock raises
    :class:`RuntimeError`. The underlying :class:`SoftReadWriteLock` handles forks, the heartbeat, and stale eviction.
    Singleton wrappers share one sync lock and its task holds.

    :param lock_file: path to the lock file; the protocol directory lives next to it as ``<lock_file>.rw``
    :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
    :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately on contention
    :param is_singleton: if ``True``, reuse existing :class:`SoftReadWriteLock` instances per resolved path
    :param heartbeat_interval: seconds between heartbeat refreshes; default 30 s
    :param stale_threshold: seconds a holder record may stay unchanged before a contender evicts it; defaults to
        ``3 * heartbeat_interval``
    :param poll_interval: seconds between acquire retries under contention; default 0.25 s
    :param on_compromise: called from the heartbeat thread with a :class:`~filelock.LeaseCompromise` when the hold is
        lost
    :param loop: event loop for ``run_in_executor``; ``None`` uses the running loop
    :param executor: executor for ``run_in_executor``; ``None`` uses the default executor

    .. versionadded:: 3.27.0

    """

    def __init__(  # ruff:ignore[too-many-arguments]  # public constructor: one parameter per documented lock option
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,
        heartbeat_interval: float = 30.0,
        stale_threshold: float | None = None,
        poll_interval: float = 0.25,
        on_compromise: Callable[[LeaseCompromise], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ) -> None:
        self._creator_pid = os.getpid()
        self._lock = SoftReadWriteLock(
            lock_file,
            timeout,
            blocking=blocking,
            is_singleton=is_singleton,
            heartbeat_interval=heartbeat_interval,
            stale_threshold=stale_threshold,
            poll_interval=poll_interval,
            on_compromise=on_compromise,
        )
        self._owners = _task_owners_for(self._lock)
        self._loop = loop
        self._executor = executor

    @property
    def lock_file(self) -> str:
        """The path to the lock file passed to the constructor."""
        return self._lock.lock_file

    @property
    def timeout(self) -> float:
        """The default timeout applied when ``acquire_read`` / ``acquire_write`` is called without one."""
        return self._lock.timeout

    @property
    def blocking(self) -> bool:
        """Whether ``acquire_*`` defaults to blocking; ``False`` makes contention raise immediately."""
        return self._lock.blocking

    @property
    def generation(self) -> int | None:
        """The generation the current hold was granted at, a fencing token; ``None`` when no lock is held."""
        return self._lock.generation

    @property
    def compromise(self) -> LeaseCompromise | None:
        """How the current hold was lost, or ``None`` while it stands."""
        return self._lock.compromise

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop used for ``run_in_executor``, or ``None`` for the running loop."""
        return self._loop

    @property
    def executor(self) -> futures.Executor | None:
        """The executor used for ``run_in_executor``, or ``None`` for the default executor."""
        return self._executor

    @asynccontextmanager
    async def read_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases a shared read lock.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :raises RuntimeError: if the calling task already holds the write lock
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        await self.acquire_read(timeout, blocking=blocking)
        try:
            yield
        finally:
            await self.release()

    @asynccontextmanager
    async def write_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases an exclusive write lock.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :raises RuntimeError: if the calling task already holds the read lock
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        await self.acquire_write(timeout, blocking=blocking)
        try:
            yield
        finally:
            await self.release()

    async def acquire_read(
        self, timeout: float | None = None, *, blocking: bool | None = None
    ) -> AsyncAcquireSoftReadWriteReturnProxy:
        """
        Acquire a shared read lock.

        See :meth:`SoftReadWriteLock.acquire_read` for reentrancy / upgrade / fork semantics. The blocking work runs
        inside ``run_in_executor`` so other coroutines on the same loop keep progressing while this call waits.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :returns: a proxy usable as an async context manager to release the lock

        :raises RuntimeError: if the calling task already holds the write lock, if this instance was invalidated by
            :func:`os.fork`, or if :meth:`close` was called
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self._raise_if_inherited()
        await self._acquire("read", timeout, blocking=blocking)
        return AsyncAcquireSoftReadWriteReturnProxy(lock=self)

    async def acquire_write(
        self, timeout: float | None = None, *, blocking: bool | None = None
    ) -> AsyncAcquireSoftReadWriteReturnProxy:
        """
        Acquire an exclusive write lock.

        See :meth:`SoftReadWriteLock.acquire_write` for the writer-preferring semantics. The blocking work runs inside
        ``run_in_executor``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        :returns: a proxy usable as an async context manager to release the lock

        :raises RuntimeError: if the calling task already holds the read lock, if this instance was invalidated by
            :func:`os.fork`, or if :meth:`close` was called
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self._raise_if_inherited()
        await self._acquire("write", timeout, blocking=blocking)
        return AsyncAcquireSoftReadWriteReturnProxy(lock=self)

    async def _acquire(self, mode: Literal["read", "write"], timeout: float | None, *, blocking: bool | None) -> None:
        blocking = self._lock.blocking if blocking is None else blocking
        sync_acquire = self._lock.acquire_read if mode == "read" else self._lock.acquire_write

        async def enter(remaining: float) -> None:
            await self._run_acquire(functools.partial(sync_acquire, remaining, blocking=blocking))

        await self._owners.acquire(
            mode,
            timeout=self._lock.timeout if timeout is None else timeout,
            blocking=blocking,
            lock_file=self.lock_file,
            enter=enter,
        )

    async def release(self, *, force: bool = False) -> None:
        """
        Release one level of the calling task's hold.

        The wrapper releases the lock once no task holds it.

        :param force: if ``True``, drop the calling task's whole hold at any nesting level

        :raises RuntimeError: if the calling task holds no lock and *force* is ``False``

        """
        if self._creator_pid == os.getpid():
            # The task table owns the nesting, so the backend is always one level deep here, and the executor can run
            # this on another worker than the acquire: force=True is the backend's cross-thread release.
            await self._owners.release(
                force=force,
                lock_file=self.lock_file,
                leave=functools.partial(self._submit, self._lock.release, force=True),
            )

    async def close(self) -> None:
        """Release any held lock and release the underlying filesystem resources. Idempotent."""
        if self._creator_pid == os.getpid():
            await self._run(self._lock.close)
            self._owners.reset()

    def _raise_if_inherited(self) -> None:
        if self._creator_pid != os.getpid():  # pragma: forked child
            msg = f"AsyncSoftReadWriteLock on {self.lock_file} was inherited across fork; construct a new instance"
            raise RuntimeError(msg)

    async def _run_acquire(self, acquire: Callable[[], AcquireReturnProxy]) -> None:
        # run_in_executor cannot recall work the pool already started, so canceling the caller does not stop the sync
        # acquire: it still commits its claim, sets the hold, and starts the heartbeat, which keeps the record fresh
        # forever so no peer on any host can evict it as stale. Wait the submitted call out and hand the claim back,
        # the way AsyncReadWriteLock does.
        acquire_future = self._submit(acquire)
        try:
            await _wait_until_done(acquire_future)
        except asyncio.CancelledError as cancellation:
            try:
                await _drain_future(acquire_future)
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            try:
                await _drain_future(self._submit(self._lock.release, force=True))
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            raise
        _future_result(acquire_future)

    async def _run(self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        # A canceled release or close is already running on the pool thread; drain it so its outcome is observed
        # instead of finishing unwatched, then let the cancellation through.
        future = self._submit(func, *args, **kwargs)
        try:
            await _wait_until_done(future)
        except asyncio.CancelledError as cancellation:
            try:
                await _drain_future(future)
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            raise
        return _future_result(future)

    def _submit(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> asyncio.Future[_BackendOutcome[_R]]:
        loop = self._loop or asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, _capture_call, functools.partial(func, *args, **kwargs))


class AsyncAcquireSoftReadWriteReturnProxy:
    """Async context-aware object that releases an :class:`AsyncSoftReadWriteLock` on exit."""

    def __init__(self, lock: AsyncSoftReadWriteLock) -> None:
        self.lock = lock

    async def __aenter__(self) -> AsyncSoftReadWriteLock:
        return self.lock

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.lock.release()


__all__ = [
    "AsyncAcquireSoftReadWriteReturnProxy",
    "AsyncSoftReadWriteLock",
]
<|endoftext|># testing.py

from contextlib import contextmanager
import re
import typing
import unittest


from .core import (
    ParserElement,
    ParseException,
    Keyword,
    __diag__,
    __compat__,
)
from . import core_builtin_exprs


class pyparsing_test:
    """
    namespace class for classes useful in writing unit tests
    """

    class reset_pyparsing_context:
        """
        Context manager to be used when writing unit tests that modify pyparsing config values:
        - packrat parsing
        - bounded recursion parsing
        - default whitespace characters
        - default keyword characters
        - literal string auto-conversion class
        - ``__diag__`` settings

        Example:

        .. testcode::

            ppt = pyparsing.pyparsing_test

            class MyTestClass(ppt.TestParseResultsAsserts):
                def test_literal(self):
                    with ppt.reset_pyparsing_context():
                        # test that literals used to construct
                        # a grammar are automatically suppressed
                        ParserElement.inline_literals_using(Suppress)

                        term = Word(alphas) | Word(nums)
                        group = Group('(' + term[...] + ')')

                        # assert that the '()' characters
                        # are not included in the parsed tokens
                        self.assertParseAndCheckList(
                            group,
                            "(abc 123 def)",
                            ['abc', '123', 'def']
                        )

                    # after exiting context manager, literals
                    # are converted to Literal expressions again
        """

        def __init__(self):
            self._save_context = {}

        def save(self):
            self._save_context["default_whitespace"] = ParserElement.DEFAULT_WHITE_CHARS
            self._save_context["default_keyword_chars"] = Keyword.DEFAULT_KEYWORD_CHARS

            self._save_context["literal_string_class"] = (
                ParserElement._literalStringClass
            )

            self._save_context["verbose_stacktrace"] = ParserElement.verbose_stacktrace

            self._save_context["packrat_enabled"] = ParserElement._packratEnabled
            if ParserElement._packratEnabled:
                self._save_context["packrat_cache_size"] = (
                    ParserElement.packrat_cache.size
                )
            else:
                self._save_context["packrat_cache_size"] = None
            self._save_context["packrat_parse"] = ParserElement._parse
            self._save_context["recursion_enabled"] = (
                ParserElement._left_recursion_enabled
            )

            self._save_context["__diag__"] = {
                name: getattr(__diag__, name) for name in __diag__._all_names
            }

            self._save_context["__compat__"] = {
                "collect_all_And_tokens": __compat__.collect_all_And_tokens
            }

            return self

        def restore(self):
            # reset pyparsing global state
            if (
                ParserElement.DEFAULT_WHITE_CHARS
                != self._save_context["default_whitespace"]
            ):
                ParserElement.set_default_whitespace_chars(
                    self._save_context["default_whitespace"]
                )

            ParserElement.verbose_stacktrace = self._save_context["verbose_stacktrace"]

            Keyword.DEFAULT_KEYWORD_CHARS = self._save_context["default_keyword_chars"]
            ParserElement.inline_literals_using(
                self._save_context["literal_string_class"]
            )

            for name, value in self._save_context["__diag__"].items():
                (__diag__.enable if value else __diag__.disable)(name)

            ParserElement._packratEnabled = False
            if self._save_context["packrat_enabled"]:
                ParserElement.enable_packrat(self._save_context["packrat_cache_size"])
            else:
                ParserElement._parse = self._save_context["packrat_parse"]
            ParserElement._left_recursion_enabled = self._save_context[
                "recursion_enabled"
            ]

            # clear debug flags on all builtins
            for expr in core_builtin_exprs:
                expr.set_debug(False)

            __compat__.collect_all_And_tokens = self._save_context["__compat__"]

            return self

        def copy(self):
            ret = type(self)()
            ret._save_context.update(self._save_context)
            return ret

        def __enter__(self):
            return self.save()

        def __exit__(self, *args):
            self.restore()

    class TestParseResultsAsserts(unittest.TestCase):
        """
        A mixin class to add parse results assertion methods to normal unittest.TestCase classes.
        """

        def assertParseResultsEquals(
            self, result, expected_list=None, expected_dict=None, msg=None
        ):
            """
            Unit test assertion to compare a :class:`ParseResults` object with an optional ``expected_list``,
            and compare any defined results names with an optional ``expected_dict``.
            """
            if expected_list is not None:
                self.assertEqual(expected_list, result.as_list(), msg=msg)
            if expected_dict is not None:
                self.assertEqual(expected_dict, result.as_dict(), msg=msg)

        def assertParseAndCheckList(
            self, expr, test_string, expected_list, msg=None, verbose=True
        ):
            """
            Convenience wrapper assert to test a parser element and input string, and assert that
            the resulting :meth:`ParseResults.as_list` is equal to the ``expected_list``.
            """
            result = expr.parse_string(test_string, parse_all=True)
            if verbose:
                print(result.dump())
            else:
                print(result.as_list())
            self.assertParseResultsEquals(result, expected_list=expected_list, msg=msg)

        def assertParseAndCheckDict(
            self, expr, test_string, expected_dict, msg=None, verbose=True
        ):
            """
            Convenience wrapper assert to test a parser element and input string, and assert that
            the resulting :meth:`ParseResults.as_dict` is equal to the ``expected_dict``.
            """
            result = expr.parse_string(test_string, parse_all=True)
            if verbose:
                print(result.dump())
            else:
                print(result.as_list())
            self.assertParseResultsEquals(result, expected_dict=expected_dict, msg=msg)

        def assertRunTestResults(
            self, run_tests_report, expected_parse_results=None, msg=None
        ):
            """
            Unit test assertion to evaluate output of
            :meth:`~ParserElement.run_tests`.

            If a list of list-dict tuples is given as the
            ``expected_parse_results`` argument, then these are zipped
            with the report tuples returned by ``run_tests()``
            and evaluated using :meth:`assertParseResultsEquals`.
            Finally, asserts that the overall
            `:meth:~ParserElement.run_tests` success value is ``True``.

            :param run_tests_report: the return value from :meth:`ParserElement.run_tests`
            :type run_tests_report: tuple[bool, list[tuple[str, ParseResults | Exception]]]
            :param expected_parse_results: (optional)
            :type expected_parse_results: list[tuple[str | list | dict | Exception, ...]]
            """
            run_test_success, run_test_results = run_tests_report

            if expected_parse_results is None:
                self.assertTrue(
                    run_test_success, msg=msg if msg is not None else "failed runTests"
                )
                return

            merged = [
                (*rpt, expected)
                for rpt, expected in zip(run_test_results, expected_parse_results)
            ]
            for test_string, result, expected in merged:
                # expected should be a tuple containing a list and/or a dict or an exception,
                # and optional failure message string
                # an empty tuple will skip any result validation
                fail_msg = next((exp for exp in expected if isinstance(exp, str)), None)
                expected_exception = next(
                    (
                        exp
                        for exp in expected
                        if isinstance(exp, type) and issubclass(exp, Exception)
                    ),
                    None,
                )
                if expected_exception is not None:
                    with self.assertRaises(
                        expected_exception=expected_exception, msg=fail_msg or msg
                    ):
                        if isinstance(result, Exception):
                            raise result
                else:
                    expected_list = next(
                        (exp for exp in expected if isinstance(exp, list)), None
                    )
                    expected_dict = next(
                        (exp for exp in expected if isinstance(exp, dict)), None
                    )
                    if (expected_list, expected_dict) != (None, None):
                        self.assertParseResultsEquals(
                            result,
                            expected_list=expected_list,
                            expected_dict=expected_dict,
                            msg=fail_msg or msg,
                        )
                    else:
                        # warning here maybe?
                        print(f"no validation for {test_string!r}")

            # do this last, in case some specific test results can be reported instead
            self.assertTrue(
                run_test_success, msg=msg if msg is not None else "failed runTests"
            )

        @contextmanager
        def assertRaisesParseException(
            self, exc_type=ParseException, expected_msg=None, msg=None
        ):
            if expected_msg is not None:
                if isinstance(expected_msg, str):
                    expected_msg = re.escape(expected_msg)
                with self.assertRaisesRegex(exc_type, expected_msg, msg=msg) as ctx:
                    yield ctx

            else:
                with self.assertRaises(exc_type, msg=msg) as ctx:
                    yield ctx

    @staticmethod
    def with_line_numbers(
        s: str,
        start_line: typing.Optional[int] = None,
        end_line: typing.Optional[int] = None,
        expand_tabs: bool = True,
        eol_mark: str = "|",
        mark_spaces: typing.Optional[str] = None,
        mark_control: typing.Optional[str] = None,
        *,
        indent: typing.Union[str, int] = "",
        base_1: bool = True,
    ) -> str:
        """
        Helpful method for debugging a parser - prints a string with line and column numbers.
        (Line and column numbers are 1-based by default - if debugging a parse action,
        pass base_1=False, to correspond to the loc value passed to the parse action.)

        :param s: string to be printed with line and column numbers
        :param start_line: starting line number in s to print (default=1)
        :param end_line: ending line number in s to print (default=len(s))
        :param expand_tabs: expand tabs to spaces, to match the pyparsing default
        :param eol_mark: string to mark the end of lines, helps visualize trailing spaces
        :param mark_spaces: special character to display in place of spaces
        :param mark_control: convert non-printing control characters to a placeholding
                             character; valid values:

                             - ``"unicode"`` - replaces control chars with Unicode symbols, such as "␍" and "␊"
                             - any single character string - replace control characters with given string
                             - ``None`` (default) - string is displayed as-is


        :param indent: string to indent with line and column numbers; if an int
                       is passed, converted to ``" " * indent``
        :param base_1: whether to label string using base 1; if False, string will be
                       labeled based at 0

        :returns: input string with leading line numbers and column number headers

        .. versionchanged:: 3.2.0
           New ``indent`` and ``base_1`` arguments.
        """
        if expand_tabs:
            s = s.expandtabs()
        if isinstance(indent, int):
            indent = " " * indent
        indent = indent.expandtabs()
        if mark_control is not None:
            mark_control = typing.cast(str, mark_control)
            if mark_control == "unicode":
                transtable_map = {
                    c: u for c, u in zip(range(0, 33), range(0x2400, 0x2433))
                }
                transtable_map[127] = 0x2421
                tbl = str.maketrans(transtable_map)
                eol_mark = ""
            else:
                ord_mark_control = ord(mark_control)
                tbl = str.maketrans(
                    {c: ord_mark_control for c in list(range(0, 32)) + [127]}
                )
            s = s.translate(tbl)
        if mark_spaces is not None and mark_spaces != " ":
            if mark_spaces == "unicode":
                tbl = str.maketrans({9: 0x2409, 32: 0x2423})
                s = s.translate(tbl)
            else:
                s = s.replace(" ", mark_spaces)
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = len(s.splitlines())
        end_line = min(end_line, len(s.splitlines()))
        start_line = min(max(0, start_line), end_line)

        if mark_control != "unicode":
            s_lines = s.splitlines()[max(start_line - base_1, 0) : end_line]
        else:
            s_lines = [
                line + "␊"
                for line in s.split("␊")[max(start_line - base_1, 0) : end_line]
            ]
        if not s_lines:
            return ""

        lineno_width = len(str(end_line))
        max_line_len = max(len(line) for line in s_lines)
        lead = f"{indent}{' ' * (lineno_width + 1)}"

        if max_line_len >= 99:
            header0 = (
                lead
                + ("" if base_1 else " ")
                + "".join(
                    f"{' ' * 99}{(i + 1) % 100}"
                    for i in range(max(max_line_len // 100, 1))
                )
                + "\n"
            )
        else:
            header0 = ""

        header1 = (
            ("" if base_1 else " ")
            + lead
            + "".join(f"         {(i + 1) % 10}" for i in range(-(-max_line_len // 10)))
            + "\n"
        )
        digits = "1234567890"
        header2 = (
            lead + ("" if base_1 else "0") + digits * (-(-max_line_len // 10)) + "\n"
        )
        return (
            header0
            + header1
            + header2
            + "\n".join(
                f"{indent}{i:{lineno_width}d}:{line}{eol_mark}"
                for i, line in enumerate(s_lines, start=start_line + base_1)
            )
            + "\n"
        )
<|endoftext|># util.py
import contextlib
import re
from functools import lru_cache, wraps
import inspect
import itertools
import types
from typing import Callable, Union, Iterable, TypeVar, cast, Any
import warnings

from .warnings import PyparsingDeprecationWarning, PyparsingDiagnosticWarning

_bslash = chr(92)
C = TypeVar("C", bound=Callable)


class __config_flags:
    """Internal class for defining compatibility and debugging flags"""

    _all_names: list[str] = []
    _fixed_names: list[str] = []
    _type_desc = "configuration"

    @classmethod
    def _set(cls, dname, value):
        if dname in cls._fixed_names:
            warnings.warn(
                f"{cls.__name__}.{dname} {cls._type_desc} is {str(getattr(cls, dname)).upper()}"
                f" and cannot be overridden",
                PyparsingDiagnosticWarning,
                stacklevel=3,
            )
            return
        if dname in cls._all_names:
            setattr(cls, dname, value)
        else:
            raise ValueError(f"no such {cls._type_desc} {dname!r}")

    enable = classmethod(lambda cls, name: cls._set(name, True))
    disable = classmethod(lambda cls, name: cls._set(name, False))


@lru_cache(maxsize=128)
def col(loc: int, strg: str) -> int:
    """
    Returns current column within a string, counting newlines as line separators.
    The first column is number 1.

    Note: the default parsing behavior is to expand tabs in the input string
    before starting the parsing process.  See
    :meth:`ParserElement.parse_string` for more
    information on parsing strings containing ``<TAB>`` s, and suggested
    methods to maintain a consistent view of the parsed string, the parse
    location, and line and column positions within the parsed string.
    """
    s = strg
    return 1 if 0 < loc < len(s) and s[loc - 1] == "\n" else loc - s.rfind("\n", 0, loc)


@lru_cache(maxsize=128)
def lineno(loc: int, strg: str) -> int:
    """Returns current line number within a string, counting newlines as line separators.
    The first line is number 1.

    Note - the default parsing behavior is to expand tabs in the input string
    before starting the parsing process.  See :meth:`ParserElement.parse_string`
    for more information on parsing strings containing ``<TAB>`` s, and
    suggested methods to maintain a consistent view of the parsed string, the
    parse location, and line and column positions within the parsed string.
    """
    return strg.count("\n", 0, loc) + 1


@lru_cache(maxsize=128)
def line(loc: int, strg: str) -> str:
    """
    Returns the line of text containing loc within a string, counting newlines as line separators.
    """
    last_cr = strg.rfind("\n", 0, loc)
    next_cr = strg.find("\n", loc)
    return strg[last_cr + 1 : next_cr] if next_cr >= 0 else strg[last_cr + 1 :]


class _UnboundedCache:
    def __init__(self):
        cache = {}
        cache_get = cache.get
        self.not_in_cache = not_in_cache = object()

        def get(_, key):
            return cache_get(key, not_in_cache)

        def set_(_, key, value):
            cache[key] = value

        def clear(_):
            cache.clear()

        self.size = None
        self.get = types.MethodType(get, self)
        self.set = types.MethodType(set_, self)
        self.clear = types.MethodType(clear, self)


class _FifoCache:
    def __init__(self, size):
        cache = {}
        self.size = size
        self.not_in_cache = not_in_cache = object()
        cache_get = cache.get
        cache_pop = cache.pop

        def get(_, key):
            return cache_get(key, not_in_cache)

        def set_(_, key, value):
            cache[key] = value
            while len(cache) > size:
                # pop oldest element in cache by getting the first key
                cache_pop(next(iter(cache)))

        def clear(_):
            cache.clear()

        self.get = types.MethodType(get, self)
        self.set = types.MethodType(set_, self)
        self.clear = types.MethodType(clear, self)


class LRUMemo:
    """
    A memoizing mapping that retains `capacity` deleted items

    The memo tracks retained items by their access order; once `capacity` items
    are retained, the least recently used item is discarded.
    """

    def __init__(self, capacity):
        self._capacity = capacity
        self._active = {}
        self._memory = {}

    def __getitem__(self, key):
        try:
            return self._active[key]
        except KeyError:
            self._memory[key] = self._memory.pop(key)
            return self._memory[key]

    def __setitem__(self, key, value):
        self._memory.pop(key, None)
        self._active[key] = value

    def __delitem__(self, key):
        try:
            value = self._active.pop(key)
        except KeyError:
            pass
        else:
            oldest_keys = list(self._memory)[: -(self._capacity + 1)]
            for key_to_delete in oldest_keys:
                self._memory.pop(key_to_delete)
            self._memory[key] = value

    def clear(self):
        self._active.clear()
        self._memory.clear()


class UnboundedMemo(dict):
    """
    A memoizing mapping that retains all deleted items
    """

    def __delitem__(self, key):
        pass


def _escape_regex_range_chars(s: str) -> str:
    # escape these chars: ^-[]
    for c in r"\^-[]":
        s = s.replace(c, _bslash + c)
    s = s.replace("\n", r"\n")
    s = s.replace("\t", r"\t")
    return str(s)


class _GroupConsecutive:
    """
    Used as a callable `key` for itertools.groupby to group
    characters that are consecutive:

    .. testcode::

       from itertools import groupby
       from pyparsing.util import _GroupConsecutive

       grouped = groupby("abcdejkmpqrs", key=_GroupConsecutive())
       for index, group in grouped:
           print(tuple([index, list(group)]))

    prints:

    .. testoutput::

       (0, ['a', 'b', 'c', 'd', 'e'])
       (1, ['j', 'k'])
       (2, ['m'])
       (3, ['p', 'q', 'r', 's'])
    """

    def __init__(self) -> None:
        self.prev = 0
        self.counter = itertools.count()
        self.value = -1

    def __call__(self, char: str) -> int:
        c_int = ord(char)
        self.prev, prev = c_int, self.prev
        if c_int - prev > 1:
            self.value = next(self.counter)
        return self.value


def _is_iterable(obj, _str_type=(str, bytes), _iter_exception=Exception):
    # str's are iterable, but in pyparsing, we don't want to iterate over them
    if isinstance(obj, _str_type):
        return False

    try:
        iter(obj)
    except _iter_exception:  # noqa
        return False
    else:
        return True


def _escape_re_range_char(c: str) -> str:
    return fr"\{c}" if c in r"\^-][" else c


def _collapse_string_to_ranges(
    s: Union[str, Iterable[str]], re_escape: bool = True
) -> str:
    r"""
    Take a string or list of single-character strings, and return
    a string of the consecutive characters in that string collapsed
    into groups, as might be used in a regular expression '[a-z]'
    character set::

        'a' -> 'a' -> '[a]'
        'bc' -> 'bc' -> '[bc]'
        'defgh' -> 'd-h' -> '[d-h]'
        'fdgeh' -> 'd-h' -> '[d-h]'
        'jklnpqrtu' -> 'j-lnp-rtu' -> '[j-lnp-rtu]'

    Duplicates get collapsed out::

        'aaa' -> 'a' -> '[a]'
        'bcbccb' -> 'bc' -> '[bc]'
        'defghhgf' -> 'd-h' -> '[d-h]'
        'jklnpqrjjjtu' -> 'j-lnp-rtu' -> '[j-lnp-rtu]'

    Spaces are preserved::

        'ab c' -> ' a-c' -> '[ a-c]'

    Characters that are significant when defining regex ranges
    get escaped::

        'acde[]-' -> r'\-\[\]ac-e' -> r'[\-\[\]ac-e]'
    """

    # Developer notes:
    # - Do not optimize this code assuming that the given input string
    #   or internal lists will be short (such as in loading generators into
    #   lists to make it easier to find the last element); this method is also
    #   used to generate regex ranges for character sets in the pyparsing.unicode
    #   classes, and these can be _very_ long lists of strings

    escape_re_range_char: Callable[[str], str]
    if re_escape:
        escape_re_range_char = _escape_re_range_char
    else:
        escape_re_range_char = lambda ss: ss

    ret = []

    # reduce input string to remove duplicates, and put in sorted order
    s_chars: list[str] = sorted(set(s))

    if len(s_chars) > 2:
        # find groups of characters that are consecutive (can be collapsed
        # down to "<first>-<last>")
        for _, chars in itertools.groupby(s_chars, key=_GroupConsecutive()):
            # _ is unimportant, is just used to identify groups
            # chars is an iterator of one or more consecutive characters
            # that comprise the current group
            first = last = next(chars)
            with contextlib.suppress(ValueError):
                *_, last = chars

            if first == last:
                # there was only a single char in this group
                ret.append(escape_re_range_char(first))

            elif last == chr(ord(first) + 1):
                # there were only 2 characters in this group
                #   'a','b' -> 'ab'
                ret.append(f"{escape_re_range_char(first)}{escape_re_range_char(last)}")

            else:
                # there were > 2 characters in this group, make into a range
                #   'c','d','e' -> 'c-e'
                ret.append(
                    f"{escape_re_range_char(first)}-{escape_re_range_char(last)}"
                )
    else:
        # only 1 or 2 chars were given to form into groups
        #   'a' -> ['a']
        #   'bc' -> ['b', 'c']
        #   'dg' -> ['d', 'g']
        # no need to list them with "-", just return as a list
        # (after escaping)
        ret = [escape_re_range_char(c) for c in s_chars]

    return "".join(ret)


def _flatten(ll: Iterable) -> list:
    ret = []
    for i in ll:
        # Developer notes:
        # - do not collapse this section of code, isinstance checks are done
        # in optimal order
        if isinstance(i, str):
            ret.append(i)
        elif isinstance(i, Iterable):
            ret.extend(_flatten(i))
        else:
            ret.append(i)
    return ret


def _convert_escaped_numerics_to_char(s: str) -> str:
    if s == "0":
        return "\0"
    if s.isdigit() and len(s) == 3:
        return chr(int(s, 8))
    elif s.startswith(("u", "x")):
        return chr(int(s[1:], 16))
    return s


def make_compressed_re(
    word_list: Iterable[str],
    max_level: int = 2,
    *,
    non_capturing_groups: bool = True,
    _level: int = 1,
) -> str:
    """
    Create a regular expression string from a list of words, collapsing by common
    prefixes and optional suffixes.

    Calls itself recursively to build nested sublists for each group of suffixes
    that have a shared prefix.
    """

    def get_suffixes_from_common_prefixes(namelist: list[str]):
        if len(namelist) > 1:
            for prefix, suffixes in itertools.groupby(namelist, key=lambda s: s[:1]):
                yield prefix, sorted([s[1:] for s in suffixes], key=len, reverse=True)
        else:
            yield namelist[0][0], [namelist[0][1:]]

    if _level == 1:
        if not word_list:
            raise ValueError("no words given to make_compressed_re()")

        if "" in word_list:
            raise ValueError("word list cannot contain empty string")
    else:
        # internal recursive call, just return empty string if no words
        if not word_list:
            return ""

    # dedupe the word list
    word_list = list({}.fromkeys(word_list))

    if max_level == 0:
        if any(len(wd) > 1 for wd in word_list):
            return "|".join(
                sorted([re.escape(wd) for wd in word_list], key=len, reverse=True)
            )
        else:
            return f"[{''.join(_escape_regex_range_chars(wd) for wd in word_list)}]"

    ret = []
    sep = ""
    ncgroup = "?:" if non_capturing_groups else ""

    for initial, suffixes in get_suffixes_from_common_prefixes(sorted(word_list)):
        ret.append(sep)
        sep = "|"

        initial = re.escape(initial)

        trailing = ""
        if "" in suffixes:
            trailing = "?"
            suffixes.remove("")

        if len(suffixes) > 1:
            if all(len(s) == 1 for s in suffixes):
                ret.append(
                    f"{initial}[{''.join(_escape_regex_range_chars(s) for s in suffixes)}]{trailing}"
                )
            else:
                if _level < max_level:
                    suffix_re = make_compressed_re(
                        sorted(suffixes),
                        max_level,
                        non_capturing_groups=non_capturing_groups,
                        _level=_level + 1,
                    )
                    ret.append(f"{initial}({ncgroup}{suffix_re}){trailing}")
                else:
                    if all(len(s) == 1 for s in suffixes):
                        ret.append(
                            f"{initial}[{''.join(_escape_regex_range_chars(s) for s in suffixes)}]{trailing}"
                        )
                    else:
                        suffixes.sort(key=len, reverse=True)
                        ret.append(
                            f"{initial}({ncgroup}{'|'.join(re.escape(s) for s in suffixes)}){trailing}"
                        )
        else:
            if suffixes:
                suffix = re.escape(suffixes[0])
                if len(suffix) > 1 and trailing:
                    ret.append(f"{initial}({ncgroup}{suffix}){trailing}")
                else:
                    ret.append(f"{initial}{suffix}{trailing}")
            else:
                ret.append(initial)
    return "".join(ret)


def replaced_by_pep8(compat_name: str, fn: C) -> C:

    # Unwrap staticmethod/classmethod
    fn = getattr(fn, "__func__", fn)

    # (Presence of 'self' arg in signature is used by explain_exception() methods, so we take
    # some extra steps to add it if present in decorated function.)
    if ["self"] == list(inspect.signature(fn).parameters)[:1]:

        @wraps(fn)
        def _inner(self, *args, **kwargs):
            warnings.warn(
                f"{compat_name!r} deprecated - use {fn.__name__!r}",
                PyparsingDeprecationWarning,
                stacklevel=2,
            )
            return fn(self, *args, **kwargs)

    else:

        @wraps(fn)
        def _inner(*args, **kwargs):
            warnings.warn(
                f"{compat_name!r} deprecated - use {fn.__name__!r}",
                PyparsingDeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

    _inner.__doc__ = f"""
        .. deprecated:: 3.0.0
           Use :class:`{fn.__name__}` instead
        """
    _inner.__name__ = compat_name
    _inner.__annotations__ = fn.__annotations__
    if isinstance(fn, types.FunctionType):
        _inner.__kwdefaults__ = fn.__kwdefaults__  # type: ignore [attr-defined]
    elif isinstance(fn, type) and hasattr(fn, "__init__"):
        _inner.__kwdefaults__ = fn.__init__.__kwdefaults__  # type: ignore [misc,attr-defined]
    else:
        _inner.__kwdefaults__ = None  # type: ignore [attr-defined]
    _inner.__qualname__ = fn.__qualname__
    return cast(C, _inner)


def _to_pep8_name(s: str, _re_sub_pattern=re.compile(r"([a-z])([A-Z])")) -> str:
    s = _re_sub_pattern.sub(r"\1_\2", s)
    return s.lower()


def deprecate_argument(
    kwargs: dict[str, Any], arg_name: str, default_value=None, *, new_name: str = ""
) -> Any:

    if arg_name in kwargs:
        new_name = new_name or _to_pep8_name(arg_name)
        warnings.warn(
            f"{arg_name!r} argument is deprecated, use {new_name!r}",
            category=PyparsingDeprecationWarning,
            stacklevel=3,
        )
    else:
        kwargs[arg_name] = default_value

    return kwargs[arg_name]
<|endoftext|># results.py

from __future__ import annotations

import collections
from collections.abc import (
    MutableMapping,
    Mapping,
    MutableSequence,
    Iterator,
    Iterable,
)
import pprint
from typing import Any

from .util import deprecate_argument, _is_iterable, _flatten


str_type: tuple[type, ...] = (str, bytes)
_generator_type = type((_ for _ in ()))
NULL_SLICE: slice = slice(None)


class _ParseResultsWithOffset:
    tup: tuple[ParseResults, int]
    __slots__ = ["tup"]

    def __init__(self, p1: ParseResults, p2: int) -> None:
        self.tup: tuple[ParseResults, int] = (p1, p2)

    def __getitem__(self, i):
        return self.tup[i]

    def __getstate__(self):
        return self.tup

    def __setstate__(self, *args):
        self.tup = args[0]


class ParseResults:
    """Structured parse results, to provide multiple means of access to
    the parsed data:

    - as a list (``len(results)``)
    - by list index (``results[0], results[1]``, etc.)
    - by attribute (``results.<results_name>`` - see :class:`ParserElement.set_results_name`)

    Example:

    .. testcode::

       integer = Word(nums)
       date_str = (integer.set_results_name("year") + '/'
                   + integer.set_results_name("month") + '/'
                   + integer.set_results_name("day"))
       # equivalent form:
       # date_str = (integer("year") + '/'
       #             + integer("month") + '/'
       #             + integer("day"))

       # parse_string returns a ParseResults object
       result = date_str.parse_string("1999/12/31")

       def test(s, fn=repr):
           print(f"{s} -> {fn(eval(s))}")

       test("list(result)")
       test("result[0]")
       test("result['month']")
       test("result.day")
       test("'month' in result")
       test("'minutes' in result")
       test("result.dump()", str)

    prints:

    .. testoutput::

       list(result) -> ['1999', '/', '12', '/', '31']
       result[0] -> '1999'
       result['month'] -> '12'
       result.day -> '31'
       'month' in result -> True
       'minutes' in result -> False
       result.dump() -> ['1999', '/', '12', '/', '31']
       - day: '31'
       - month: '12'
       - year: '1999'

    """

    _null_values: tuple[Any, ...] = (None, [], ())

    _name: str
    _parent: ParseResults
    _all_names: set[str]
    _modal: bool
    _toklist: list[Any]
    _tokdict: dict[str, Any]

    __slots__ = (
        "_name",
        "_parent",
        "_all_names",
        "_modal",
        "_toklist",
        "_tokdict",
    )

    class List(list):
        """
        Simple wrapper class to distinguish parsed list results that should be preserved
        as actual Python lists, instead of being converted to :class:`ParseResults`:

        .. testcode::

           import pyparsing as pp
           ppc = pp.common

           LBRACK, RBRACK, LPAR, RPAR = pp.Suppress.using_each("[]()")
           element = pp.Forward()
           item = ppc.integer
           item_list = pp.DelimitedList(element)
           element_list = LBRACK + item_list + RBRACK | LPAR + item_list + RPAR
           element <<= item | element_list

           # add parse action to convert from ParseResults
           # to actual Python collection types
           @element_list.add_parse_action
           def as_python_list(t):
               return pp.ParseResults.List(t.as_list())

           element.run_tests('''
               100
               [2,3,4]
               [[2, 1],3,4]
               [(2, 1),3,4]
               (2,3,4)
               ([2, 3], 4)
               ''', post_parse=lambda s, r: (r[0], type(r[0]))
           )

        prints:

        .. testoutput::
           :options: +NORMALIZE_WHITESPACE


           100
           (100, <class 'int'>)

           [2,3,4]
           ([2, 3, 4], <class 'list'>)

           [[2, 1],3,4]
           ([[2, 1], 3, 4], <class 'list'>)

           [(2, 1),3,4]
           ([[2, 1], 3, 4], <class 'list'>)

           (2,3,4)
           ([2, 3, 4], <class 'list'>)

           ([2, 3], 4)
           ([[2, 3], 4], <class 'list'>)

        (Used internally by :class:`Group` when `aslist=True`.)
        """

        def __new__(cls, contained=None):
            if contained is None:
                contained = []

            if not isinstance(contained, list):
                raise TypeError(
                    f"{cls.__name__} may only be constructed with a list, not {type(contained).__name__}"
                )

            return list.__new__(cls)

    def __new__(cls, toklist=None, name=None, **kwargs):
        if isinstance(toklist, ParseResults):
            return toklist
        self = object.__new__(cls)
        self._name = None
        self._parent = None
        self._all_names = set()

        if toklist is None:
            self._toklist = []
        elif isinstance(toklist, (list, _generator_type)):
            self._toklist = (
                [toklist[:]]
                if isinstance(toklist, ParseResults.List)
                else list(toklist)
            )
        else:
            self._toklist = [toklist]
        self._tokdict = dict()
        return self

    # Performance tuning: we construct a *lot* of these, so keep this
    # constructor as small and fast as possible
    def __init__(
        self,
        toklist=None,
        name=None,
        aslist=True,
        modal=True,
        isinstance=isinstance,
        **kwargs,
    ) -> None:
        asList = deprecate_argument(kwargs, "asList", True, new_name="aslist")

        asList = asList and aslist
        self._tokdict: dict[str, _ParseResultsWithOffset]
        self._modal = modal

        if name is None or name == "":
            return

        if isinstance(name, int):
            name = str(name)

        if not modal:
            self._all_names = {name}

        self._name = name

        if toklist in self._null_values:
            return

        if isinstance(toklist, (str_type, type)):
            toklist = [toklist]

        if asList:
            if isinstance(toklist, ParseResults):
                self[name] = _ParseResultsWithOffset(ParseResults(toklist._toklist), 0)
            else:
                self[name] = _ParseResultsWithOffset(ParseResults(toklist[0]), 0)
            self[name]._name = name
            return

        try:
            self[name] = toklist[0]
        except (KeyError, TypeError, IndexError):
            if toklist is not self:
                self[name] = toklist
            else:
                self._name = name

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self._toklist[i]

        if i not in self._all_names:
            return self._tokdict[i][-1][0]

        return ParseResults([v[0] for v in self._tokdict[i]])

    def __setitem__(self, k, v, isinstance=isinstance):
        if isinstance(v, _ParseResultsWithOffset):
            self._tokdict[k] = self._tokdict.get(k, list()) + [v]
            sub = v[0]
        elif isinstance(k, (int, slice)):
            self._toklist[k] = v
            sub = v
        else:
            self._tokdict[k] = self._tokdict.get(k, []) + [
                _ParseResultsWithOffset(v, 0)
            ]
            sub = v
        if isinstance(sub, ParseResults):
            sub._parent = self

    def __delitem__(self, i):
        if not isinstance(i, (int, slice)):
            del self._tokdict[i]
            return

        # slight optimization if del results[:]
        if i == NULL_SLICE:
            self._toklist.clear()
            return

        mylen = len(self._toklist)
        del self._toklist[i]

        # convert int to slice
        if isinstance(i, int):
            if i < 0:
                i += mylen
            i = slice(i, i + 1)
        # get removed indices
        removed = list(range(*i.indices(mylen)))
        removed.reverse()
        # fixup indices in token dictionary
        for occurrences in self._tokdict.values():
            for j in removed:
                for k, (value, position) in enumerate(occurrences):
                    occurrences[k] = _ParseResultsWithOffset(
                        value, position - (position > j)
                    )

    def __contains__(self, k) -> bool:
        return k in self._tokdict

    def __len__(self) -> int:
        return len(self._toklist)

    def __bool__(self) -> bool:
        return not not (self._toklist or self._tokdict)

    def __iter__(self) -> Iterator:
        return iter(self._toklist)

    def __reversed__(self) -> Iterator:
        return iter(self._toklist[::-1])

    def keys(self):
        return iter(self._tokdict)

    def values(self):
        return (self[k] for k in self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def haskeys(self) -> bool:
        """
        Since ``keys()`` returns an iterator, this method is helpful in bypassing
        code that looks for the existence of any defined results names."""
        return not not self._tokdict

    def pop(self, *args, **kwargs):
        """
        Removes and returns item at specified index (default= ``last``).
        Supports both ``list`` and ``dict`` semantics for ``pop()``. If
        passed no argument or an integer argument, it will use ``list``
        semantics and pop tokens from the list of parsed tokens. If passed
        a non-integer argument (most likely a string), it will use ``dict``
        semantics and pop the corresponding value from any defined results
        names. A second default return value argument is supported, just as in
        ``dict.pop()``.

        Example:

        .. doctest::

           >>> numlist = Word(nums)[...]
           >>> print(numlist.parse_string("0 123 321"))
           ['0', '123', '321']

           >>> def remove_first(tokens):
           ...     tokens.pop(0)
           ...
           >>> numlist.add_parse_action(remove_first)
           [W:(0-9)]...
           >>> print(numlist.parse_string("0 123 321"))
           ['123', '321']

           >>> label = Word(alphas)
           >>> patt = label("LABEL") + Word(nums)[1, ...]
           >>> print(patt.parse_string("AAB 123 321").dump())
           ['AAB', '123', '321']
           - LABEL: 'AAB'

           >>> # Use pop() in a parse action to remove named result
           >>> # (note that corresponding value is not
           >>> # removed from list form of results)
           >>> def remove_LABEL(tokens):
           ...     tokens.pop("LABEL")
           ...     return tokens
           ...
           >>> patt.add_parse_action(remove_LABEL)
           {W:(A-Za-z) {W:(0-9)}...}
           >>> print(patt.parse_string("AAB 123 321").dump())
           ['AAB', '123', '321']

        """
        if not args:
            args = [-1]
        for k, v in kwargs.items():
            if k == "default":
                args = (args[0], v)
            else:
                raise TypeError(f"pop() got an unexpected keyword argument {k!r}")
        if isinstance(args[0], int) or len(args) == 1 or args[0] in self:
            index = args[0]
            ret = self[index]
            del self[index]
            return ret
        else:
            defaultvalue = args[1]
            return defaultvalue

    def get(self, key, default_value=None):
        """
        Returns named result matching the given key, or if there is no
        such name, then returns the given ``default_value`` or ``None`` if no
        ``default_value`` is specified.

        Similar to ``dict.get()``.

        Example:

        .. doctest::

           >>> integer = Word(nums)
           >>> date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

           >>> result = date_str.parse_string("1999/12/31")
           >>> result.get("year")
           '1999'
           >>> result.get("hour", "not specified")
           'not specified'
           >>> result.get("hour")

        """
        if key in self:
            return self[key]
        else:
            return default_value

    def insert(self, index, ins_string):
        """
        Inserts new element at location index in the list of parsed tokens.

        Similar to ``list.insert()``.

        Example:

        .. doctest::

           >>> numlist = Word(nums)[...]
           >>> print(numlist.parse_string("0 123 321"))
           ['0', '123', '321']

           >>> # use a parse action to insert the parse location
           >>> # in the front of the parsed results
           >>> def insert_locn(locn, tokens):
           ...     tokens.insert(0, locn)
           ...
           >>> numlist.add_parse_action(insert_locn)
           [W:(0-9)]...
           >>> print(numlist.parse_string("0 123 321"))
           [0, '0', '123', '321']

        """
        self._toklist.insert(index, ins_string)
        # fixup indices in token dictionary
        for occurrences in self._tokdict.values():
            for k, (value, position) in enumerate(occurrences):
                occurrences[k] = _ParseResultsWithOffset(
                    value, position + (position > index)
                )

    def append(self, item):
        """
        Add single element to end of ``ParseResults`` list of elements.

        Example:

        .. doctest::

           >>> numlist = Word(nums)[...]
           >>> print(numlist.parse_string("0 123 321"))
           ['0', '123', '321']

           >>> # use a parse action to compute the sum of the parsed integers,
           >>> # and add it to the end
           >>> def append_sum(tokens):
           ...     tokens.append(sum(map(int, tokens)))
           ...
           >>> numlist.add_parse_action(append_sum)
           [W:(0-9)]...
           >>> print(numlist.parse_string("0 123 321"))
           ['0', '123', '321', 444]
        """
        self._toklist.append(item)

    def extend(self, itemseq):
        """
        Add sequence of elements to end of :class:`ParseResults` list of elements.

        Example:

        .. testcode::

           patt = Word(alphas)[1, ...]

           # use a parse action to append the reverse of the matched strings,
           # to make a palindrome
           def make_palindrome(tokens):
               tokens.extend(reversed([t[::-1] for t in tokens]))
               return ''.join(tokens)

           patt.add_parse_action(make_palindrome)
           print(patt.parse_string("lskdj sdlkjf lksd"))

        prints:

        .. testoutput::

           ['lskdjsdlkjflksddsklfjkldsjdksl']
        """
        if isinstance(itemseq, ParseResults):
            self.__iadd__(itemseq)
        else:
            self._toklist.extend(itemseq)

    def clear(self):
        """
        Clear all elements and results names.
        """
        del self._toklist[:]
        self._tokdict.clear()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            if name.startswith("__"):
                raise AttributeError(name)
            return ""

    def __add__(self, other: ParseResults) -> ParseResults:
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other: ParseResults) -> ParseResults:
        if not other:
            return self

        if other._tokdict:
            offset = len(self._toklist)
            addoffset = lambda a: offset if a < 0 else a + offset
            otheritems = other._tokdict.items()
            otherdictitems = [
                (k, _ParseResultsWithOffset(v[0], addoffset(v[1])))
                for k, vlist in otheritems
                for v in vlist
            ]
            for k, v in otherdictitems:
                self[k] = v
                if isinstance(v[0], ParseResults):
                    v[0]._parent = self

        self._toklist += other._toklist
        self._all_names |= other._all_names
        return self

    def __radd__(self, other) -> ParseResults:
        if isinstance(other, int) and other == 0:
            # useful for merging many ParseResults using sum() builtin
            return self.copy()
        else:
            # this may raise a TypeError - so be it
            return other + self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._toklist!r}, {self.as_dict()})"

    def __str__(self) -> str:
        return (
            "["
            + ", ".join(
                [
                    str(i) if isinstance(i, ParseResults) else repr(i)
                    for i in self._toklist
                ]
            )
            + "]"
        )

    def _asStringList(self, sep=""):
        out = []
        for item in self._toklist:
            if out and sep:
                out.append(sep)
            if isinstance(item, ParseResults):
                out += item._asStringList()
            else:
                out.append(str(item))
        return out

    def as_list(self, *, flatten: bool = False) -> list:
        """
        Returns the parse results as a nested list of matching tokens, all converted to strings.
        If ``flatten`` is True, all the nesting levels in the returned list are collapsed.

        Example:

        .. doctest::

           >>> patt = Word(alphas)[1, ...]
           >>> result = patt.parse_string("sldkj lsdkj sldkj")
           >>> # even though the result prints in string-like form,
           >>> # it is actually a pyparsing ParseResults
           >>> type(result)
           <class 'pyparsing.results.ParseResults'>
           >>> print(result)
           ['sldkj', 'lsdkj', 'sldkj']

        .. doctest::

           >>> # Use as_list() to create an actual list
           >>> result_list = result.as_list()
           >>> type(result_list)
           <class 'list'>
           >>> print(result_list)
           ['sldkj', 'lsdkj', 'sldkj']

        .. versionchanged:: 3.2.0
           New ``flatten`` argument.
        """

        if flatten:
            return [*_flatten(self)]
        else:
            return [
                res.as_list() if isinstance(res, ParseResults) else res
                for res in self._toklist
            ]

    def as_dict(self) -> dict:
        """
        Returns the named parse results as a nested dictionary.

        Example:

        .. doctest::

           >>> integer = pp.Word(pp.nums)
           >>> date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

           >>> result = date_str.parse_string('1999/12/31')
           >>> type(result)
           <class 'pyparsing.results.ParseResults'>
           >>> result
           ParseResults(['1999', '/', '12', '/', '31'], {'year': '1999', 'month': '12', 'day': '31'})

           >>> result_dict = result.as_dict()
           >>> type(result_dict)
           <class 'dict'>
           >>> result_dict
           {'year': '1999', 'month': '12', 'day': '31'}

           >>> # even though a ParseResults supports dict-like access,
           >>> # sometime you just need to have a dict
           >>> import json
           >>> print(json.dumps(result))
           Traceback (most recent call last):
           TypeError: Object of type ParseResults is not JSON serializable
           >>> print(json.dumps(result.as_dict()))
           {"year": "1999", "month": "12", "day": "31"}
        """

        def to_item(obj):
            if isinstance(obj, ParseResults):
                return obj.as_dict() if obj.haskeys() else [to_item(v) for v in obj]
            else:
                return obj

        return dict((k, to_item(v)) for k, v in self.items())

    def copy(self) -> ParseResults:
        """
        Returns a new shallow copy of a :class:`ParseResults` object.
        :class:`ParseResults` items contained within the source are
        shared with the copy. Use :meth:`ParseResults.deepcopy` to
        create a copy with its own separate content values.
        """
        ret = ParseResults(self._toklist)
        ret._tokdict = self._tokdict.copy()
        ret._parent = self._parent
        ret._all_names |= self._all_names
        ret._name = self._name
        return ret

    def deepcopy(self) -> ParseResults:
        """
        Returns a new deep copy of a :class:`ParseResults` object.

        .. versionadded:: 3.1.0
        """
        ret = self.copy()
        # replace values with copies if they are of known mutable types
        for i, obj in enumerate(self._toklist):
            if isinstance(obj, ParseResults):
                ret._toklist[i] = obj.deepcopy()
            elif isinstance(obj, (str, bytes)):
                pass
            elif isinstance(obj, MutableMapping):
                ret._toklist[i] = dest = type(obj)()
                for k, v in obj.items():
                    dest[k] = v.deepcopy() if isinstance(v, ParseResults) else v
            elif isinstance(obj, Iterable):
                ret._toklist[i] = type(obj)(
                    v.deepcopy() if isinstance(v, ParseResults) else v for v in obj  # type: ignore[call-arg]
                )
        return ret

    def get_name(self) -> str | None:
        r"""
        Returns the results name for this token expression.

        Useful when several different expressions might match
        at a particular location.

        Example:

        .. testcode::

           integer = Word(nums)
           ssn_expr = Regex(r"\d\d\d-\d\d-\d\d\d\d")
           house_number_expr = Suppress('#') + Word(nums, alphanums)
           user_data = (Group(house_number_expr)("house_number")
                       | Group(ssn_expr)("ssn")
                       | Group(integer)("age"))
           user_info = user_data[1, ...]

           result = user_info.parse_string("22 111-22-3333 #221B")
           for item in result:
               print(item.get_name(), ':', item[0])

        prints:

        .. testoutput::

           age : 22
           ssn : 111-22-3333
           house_number : 221B

        """
        if self._name:
            return self._name
        elif self._parent:
            par: ParseResults = self._parent
            parent_tokdict_items = par._tokdict.items()
            return next(
                (
                    k
                    for k, vlist in parent_tokdict_items
                    for v, loc in vlist
                    if v is self
                ),
                None,
            )
        elif (
            len(self) == 1
            and len(self._tokdict) == 1
            and next(iter(self._tokdict.values()))[0][1] in (0, -1)
        ):
            return next(iter(self._tokdict.keys()))
        else:
            return None

    def dump(self, indent="", full=True, include_list=True, _depth=0) -> str:
        """
        Diagnostic method for listing out the contents of
        a :class:`ParseResults`. Accepts an optional ``indent`` argument so
        that this string can be embedded in a nested display of other data.

        Example:

        .. testcode::

           integer = Word(nums)
           date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

           result = date_str.parse_string('1999/12/31')
           print(result.dump())

        prints:

        .. testoutput::

           ['1999', '/', '12', '/', '31']
           - day: '31'
           - month: '12'
           - year: '1999'
        """
        out = []
        NL = "\n"
        out.append(indent + str(self.as_list()) if include_list else "")

        if not full:
            return "".join(out)

        if self.haskeys():
            items = sorted((str(k), v) for k, v in self.items())
            for k, v in items:
                if out:
                    out.append(NL)
                out.append(f"{indent}{('  ' * _depth)}- {k}: ")
                if not isinstance(v, ParseResults):
                    out.append(repr(v))
                    continue

                if not v:
                    out.append(str(v))
                    continue

                out.append(
                    v.dump(
                        indent=indent,
                        full=full,
                        include_list=include_list,
                        _depth=_depth + 1,
                    )
                )
        if not any(isinstance(vv, ParseResults) for vv in self):
            return "".join(out)

        v = self
        incr = "  "
        nl = "\n"
        for i, vv in enumerate(v):
            if isinstance(vv, ParseResults):
                vv_dump = vv.dump(
                    indent=indent,
                    full=full,
                    include_list=include_list,
                    _depth=_depth + 1,
                )
                out.append(
                    f"{nl}{indent}{incr * _depth}[{i}]:{nl}{indent}{incr * (_depth + 1)}{vv_dump}"
                )
            else:
                out.append(
                    f"{nl}{indent}{incr * _depth}[{i}]:{nl}{indent}{incr * (_depth + 1)}{vv}"
                )

        return "".join(out)

    def pprint(self, *args, **kwargs):
        """
        Pretty-printer for parsed results as a list, using the
        `pprint <https://docs.python.org/3/library/pprint.html>`_ module.
        Accepts additional positional or keyword args as defined for
        `pprint.pprint <https://docs.python.org/3/library/pprint.html#pprint.pprint>`_ .

        Example:

        .. testcode::

           ident = Word(alphas, alphanums)
           num = Word(nums)
           func = Forward()
           term = ident | num | Group('(' + func + ')')
           func <<= ident + Group(Optional(DelimitedList(term)))
           result = func.parse_string("fna a,b,(fnb c,d,200),100")
           result.pprint(width=40)

        prints:

        .. testoutput::

           ['fna',
            ['a',
             'b',
             ['(', 'fnb', ['c', 'd', '200'], ')'],
             '100']]
        """
        pprint.pprint(self.as_list(), *args, **kwargs)

    # add support for pickle protocol
    def __getstate__(self):
        return (
            self._toklist,
            (
                self._tokdict.copy(),
                None,
                self._all_names,
                self._name,
            ),
        )

    def __setstate__(self, state):
        self._toklist, (self._tokdict, par, inAccumNames, self._name) = state
        self._all_names = set(inAccumNames)
        self._parent = None

    def __getnewargs__(self):
        return self._toklist, self._name

    def __dir__(self):
        return dir(type(self)) + list(self.keys())

    @classmethod
    def from_dict(cls, other, name=None) -> ParseResults:
        """
        Helper classmethod to construct a :class:`ParseResults` from a ``dict``, preserving the
        name-value relations as results names. If an optional ``name`` argument is
        given, a nested :class:`ParseResults` will be returned.
        """
        ret = cls([])
        for k, v in other.items():
            if isinstance(v, Mapping):
                ret += cls.from_dict(v, name=k)
            else:
                ret += cls([v], name=k, aslist=_is_iterable(v))
        if name is not None:
            ret = cls([ret], name=name)
        return ret

    asList = as_list
    """
    .. deprecated:: 3.0.0
       use :meth:`as_list`
    """
    asDict = as_dict
    """
    .. deprecated:: 3.0.0
       use :meth:`as_dict`
    """
    getName = get_name
    """
    .. deprecated:: 3.0.0
       use :meth:`get_name`
    """


MutableMapping.register(ParseResults)
MutableSequence.register(ParseResults)
<|endoftext|># common.py
from .core import *
from .helpers import DelimitedList, any_open_tag, any_close_tag
from datetime import datetime
import sys

PY_310_OR_LATER = sys.version_info >= (3, 10)


# some other useful expressions - using lower-case class name since we are really using this as a namespace
class pyparsing_common:
    """Here are some common low-level expressions that may be useful in
    jump-starting parser development:

    - numeric forms (:class:`integers<integer>`, :class:`reals<real>`,
      :class:`scientific notation<sci_real>`)
    - common :class:`programming identifiers<identifier>`
    - network addresses (:class:`MAC<mac_address>`,
      :class:`IPv4<ipv4_address>`, :class:`IPv6<ipv6_address>`)
    - ISO8601 :class:`dates<iso8601_date>` and
      :class:`datetime<iso8601_datetime>`
    - :class:`UUID<uuid>`
    - :class:`comma-separated list<comma_separated_list>`
    - :class:`url`

    Parse actions:

    - :class:`convert_to_integer`
    - :class:`convert_to_float`
    - :class:`convert_to_date`
    - :class:`convert_to_datetime`
    - :class:`strip_html_tags`
    - :class:`upcase_tokens`
    - :class:`downcase_tokens`

    Examples:

    .. testcode::

        pyparsing_common.number.run_tests('''
            # any int or real number, returned as the appropriate type
            100
            -100
            +100
            3.14159
            6.02e23
            1e-12
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # any int or real number, returned as the appropriate type
        100
        [100]

        -100
        [-100]

        +100
        [100]

        3.14159
        [3.14159]

        6.02e23
        [6.02e+23]

        1e-12
        [1e-12]

    .. testcode::

        pyparsing_common.fnumber.run_tests('''
            # any int or real number, returned as float
            100
            -100
            +100
            3.14159
            6.02e23
            1e-12
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # any int or real number, returned as float
        100
        [100.0]

        -100
        [-100.0]

        +100
        [100.0]

        3.14159
        [3.14159]

        6.02e23
        [6.02e+23]

        1e-12
        [1e-12]

    .. testcode::

        pyparsing_common.hex_integer.run_tests('''
            # hex numbers
            100
            FF
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # hex numbers
        100
        [256]

        FF
        [255]

    .. testcode::

        pyparsing_common.fraction.run_tests('''
            # fractions
            1/2
            -3/4
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # fractions
        1/2
        [0.5]

        -3/4
        [-0.75]

    .. testcode::

        pyparsing_common.mixed_integer.run_tests('''
            # mixed fractions
            1
            1/2
            -3/4
            1-3/4
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # mixed fractions
        1
        [1]

        1/2
        [0.5]

        -3/4
        [-0.75]

        1-3/4
        [1.75]
    .. testcode::

        import uuid
        pyparsing_common.uuid.set_parse_action(token_map(uuid.UUID))
        pyparsing_common.uuid.run_tests('''
            # uuid
            12345678-1234-5678-1234-567812345678
            ''')

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE


        # uuid
        12345678-1234-5678-1234-567812345678
        [UUID('12345678-1234-5678-1234-567812345678')]
    """

    @staticmethod
    def convert_to_integer(_, __, t):
        """
        Parse action for converting parsed integers to Python int
        """
        return [int(tt) for tt in t]

    @staticmethod
    def convert_to_float(_, __, t):
        """
        Parse action for converting parsed numbers to Python float
        """
        return [float(tt) for tt in t]

    integer = (
        Word(nums)
        .set_name("integer")
        .set_parse_action(
            convert_to_integer
            if PY_310_OR_LATER
            else lambda t: [int(tt) for tt in t]
        )
    )
    """expression that parses an unsigned integer, converts to an int"""

    hex_integer = (
        Word(hexnums).set_name("hex integer").set_parse_action(token_map(int, 16))
    )
    """expression that parses a hexadecimal integer, converts to an int"""

    signed_integer = (
        Regex(r"[+-]?\d+")
        .set_name("signed integer")
        .set_parse_action(
            convert_to_integer
            if PY_310_OR_LATER
            else lambda t: [int(tt) for tt in t]
        )
    )
    """expression that parses an integer with optional leading sign, converts to an int"""

    fraction = (
        signed_integer().set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
        + "/"
        + signed_integer().set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
    ).set_name("fraction")
    """fractional expression of an integer divided by an integer, converts to a float"""
    fraction.add_parse_action(lambda tt: tt[0] / tt[-1])

    mixed_integer = (
        fraction | signed_integer + Opt(Opt("-").suppress() + fraction)
    ).set_name("fraction or mixed integer-fraction")
    """mixed integer of the form 'integer - fraction', with optional leading integer, converts to a float"""
    mixed_integer.add_parse_action(sum)

    real = (
        Regex(r"[+-]?(?:\d+\.\d*|\.\d+)")
        .set_name("real number")
        .set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
    )
    """expression that parses a floating point number, converts to a float"""

    sci_real = (
        Regex(r"[+-]?(?:\d+(?:[eE][+-]?\d+)|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?)")
        .set_name("real number with scientific notation")
        .set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
    )
    """expression that parses a floating point number with optional
    scientific notation, converts to a float"""

    # streamlining this expression makes the docs nicer-looking
    number = (sci_real | real | signed_integer).set_name("number").streamline()
    """any numeric expression, converts to the corresponding Python type"""

    fnumber = (
        Regex(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?")
        .set_name("fnumber")
        .set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
    )
    """any int or real number, always converts to a float"""

    ieee_float = (
        Regex(r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))")
        .set_name("ieee_float")
        .set_parse_action(
            convert_to_float
            if PY_310_OR_LATER
            else lambda t: [float(tt) for tt in t]
        )
    )
    """any floating-point literal (int, real number, infinity, or NaN), converts to a float"""

    identifier = Word(identchars, identbodychars).set_name("identifier")
    """typical code identifier (leading alpha or '_', followed by 0 or more alphas, nums, or '_')"""

    ipv4_address = Regex(
        r"(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}"
    ).set_name("IPv4 address")
    "IPv4 address (``0.0.0.0 - 255.255.255.255``)"

    _ipv6_part = Regex(r"[0-9a-fA-F]{1,4}").set_name("hex_integer")
    _full_ipv6_address = (_ipv6_part + (":" + _ipv6_part) * 7).set_name(
        "full IPv6 address"
    )
    _short_ipv6_address = (
        Opt(_ipv6_part + (":" + _ipv6_part) * (0, 6))
        + "::"
        + Opt(_ipv6_part + (":" + _ipv6_part) * (0, 6))
    ).set_name("short IPv6 address")
    _short_ipv6_address.add_condition(
        lambda t: sum(1 for tt in t if pyparsing_common._ipv6_part.matches(tt)) < 8
    )
    _mixed_ipv6_address = ("::ffff:" + ipv4_address).set_name("mixed IPv6 address")
    ipv6_address = Combine(
        (_full_ipv6_address | _mixed_ipv6_address | _short_ipv6_address).set_name(
            "IPv6 address"
        )
    ).set_name("IPv6 address")
    "IPv6 address (long, short, or mixed form)"

    mac_address = Regex(
        r"[0-9a-fA-F]{2}([:.-])[0-9a-fA-F]{2}(?:\1[0-9a-fA-F]{2}){4}"
    ).set_name("MAC address")
    "MAC address xx:xx:xx:xx:xx (may also have '-' or '.' delimiters)"

    @staticmethod
    def convert_to_date(fmt: str = "%Y-%m-%d"):
        """
        Helper to create a parse action for converting parsed date string to Python datetime.date

        Params -
        - fmt - format to be passed to datetime.strptime (default= ``"%Y-%m-%d"``)

        Example:

        .. testcode::

            date_expr = pyparsing_common.iso8601_date.copy()
            date_expr.set_parse_action(pyparsing_common.convert_to_date())
            print(date_expr.parse_string("1999-12-31"))

        prints:

        .. testoutput::

            [datetime.date(1999, 12, 31)]
        """

        def cvt_fn(ss, ll, tt):
            try:
                return datetime.strptime(tt[0], fmt).date()
            except ValueError as ve:
                raise ParseException(ss, ll, str(ve))

        return cvt_fn

    @staticmethod
    def convert_to_datetime(fmt: str = "%Y-%m-%dT%H:%M:%S.%f"):
        """Helper to create a parse action for converting parsed
        datetime string to Python :class:`datetime.datetime`

        Params -
        - fmt - format to be passed to :class:`datetime.strptime` (default= ``"%Y-%m-%dT%H:%M:%S.%f"``)

        Example:

        .. testcode::

            dt_expr = pyparsing_common.iso8601_datetime.copy()
            dt_expr.set_parse_action(pyparsing_common.convert_to_datetime())
            print(dt_expr.parse_string("1999-12-31T23:59:59.999"))

        prints:

        .. testoutput::

            [datetime.datetime(1999, 12, 31, 23, 59, 59, 999000)]
        """

        def cvt_fn(s, l, t):
            try:
                return datetime.strptime(t[0], fmt)
            except ValueError as ve:
                raise ParseException(s, l, str(ve))

        return cvt_fn

    iso8601_date = Regex(
        r"(?P<year>\d{4})(?:-(?P<month>\d\d)(?:-(?P<day>\d\d))?)?"
    ).set_name("ISO8601 date")
    "ISO8601 date (``yyyy-mm-dd``)"

    iso8601_datetime = Regex(
        r"(?P<year>\d{4})-(?P<month>\d\d)-(?P<day>\d\d)[T ](?P<hour>\d\d):(?P<minute>\d\d)(:(?P<second>\d\d(\.\d*)?)?)?(?P<tz>Z|[+-]\d\d:?\d\d)?"
    ).set_name("ISO8601 datetime")
    "ISO8601 datetime (``yyyy-mm-ddThh:mm:ss.s(Z|+-00:00)``) - trailing seconds, milliseconds, and timezone optional; accepts separating ``'T'`` or ``' '``"

    @staticmethod
    def as_datetime(s, l, t):
        """Parse action to convert parsed dates or datetimes to a Python
        :class:`datetime.datetime`.

        This parse action will use the year, month, day, etc. results
        names defined in the ISO8601 date expressions, but it can be
        used with any expression that provides one or more of these fields.

        Omitted fields will default to fields from Jan 1, 00:00:00.

        Invalid dates will raise a :class:`ParseException` with the
        error message indicating the invalid date fields.
        """
        year = int(t.year.lstrip("0") or 0)
        month = int(t.month or 1)
        day = int(t.day or 1)
        hour = int(t.hour or 0)
        minute = int(t.minute or 0)
        second = float(t.second or 0)
        try:
            return datetime(
                year, month, day, hour, minute, int(second), int((second % 1) * 1000)
            )
        except ValueError as ve:
            raise ParseException(t, l, f"Invalid date/time: {ve}").with_traceback(
                ve.__traceback__
            ) from None

    if PY_310_OR_LATER:
        iso8601_date_validated = iso8601_date().add_parse_action(as_datetime)
        "Validated ISO8601 date strings, raising :class:`ParseException` for invalid date values."

        iso8601_datetime_validated = iso8601_datetime().add_parse_action(as_datetime)
        "Validated ISO8601 date and time strings, raising :class:`ParseException` for invalid date/time values."

    uuid = Regex(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}").set_name(
        "UUID"
    )
    "UUID (``xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx``)"

    _html_stripper = any_open_tag.suppress() | any_close_tag.suppress()

    @staticmethod
    def strip_html_tags(s: str, l: int, tokens: ParseResults):
        """Parse action to remove HTML tags from web page HTML source

        Example:

        .. testcode::

            # strip HTML links from normal text
            text = '<td>More info at the <a href="https://github.com/pyparsing/pyparsing/wiki">pyparsing</a> wiki page</td>'
            td, td_end = make_html_tags("TD")
            table_text = td + SkipTo(td_end).set_parse_action(
                pyparsing_common.strip_html_tags)("body") + td_end
            print(table_text.parse_string(text).body)

        Prints:

        .. testoutput::

            More info at the pyparsing wiki page
        """
        return pyparsing_common._html_stripper.transform_string(tokens[0])

    _commasepitem = (
        Combine(
            OneOrMore(
                ~Literal(",")
                + ~LineEnd()
                + Word(printables, exclude_chars=",")
                + Opt(White(" \t") + ~FollowedBy(LineEnd() | ","))
            )
        )
        .streamline()
        .set_name("commaItem")
    )
    comma_separated_list = DelimitedList(
        Opt(quoted_string.copy() | _commasepitem, default="")
    ).set_name("comma separated list")
    """Predefined expression of 1 or more printable words or quoted strings, separated by commas."""

    @staticmethod
    def upcase_tokens(s, l, t):
        """Parse action to convert tokens to upper case."""
        return [tt.upper() for tt in t]

    @staticmethod
    def downcase_tokens(s, l, t):
        """Parse action to convert tokens to lower case."""
        return [tt.lower() for tt in t]

    # fmt: off
    url = Regex(
        # https://mathiasbynens.be/demo/url-regex
        # https://gist.github.com/dperini/729294
        r"(?P<url>"
        # protocol identifier (optional)
        # short syntax // still required
        r"(?:(?:(?P<scheme>https?|ftp):)?\/\/)"
        # user:pass BasicAuth (optional)
        r"(?:(?P<auth>\S+(?::\S*)?)@)?"
        r"(?P<host>"
        # IP address exclusion
        # private & local networks
        r"(?!(?:10|127)(?:\.\d{1,3}){3})"
        r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
        r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
        # IP address dotted notation octets
        # excludes loopback network 0.0.0.0
        # excludes reserved space >= 224.0.0.0
        # excludes network & broadcast addresses
        # (first & last IP address of each class)
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
        r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
        r"|"
        # host & domain names, may end with dot
        # can be replaced by a shortest alternative
        # (?![-_])(?:[-\w\u00a1-\uffff]{0,63}[^-_]\.)+
        r"(?:"
        r"(?:"
        r"[a-z0-9\u00a1-\uffff]"
        r"[a-z0-9\u00a1-\uffff_-]{0,62}"
        r")?"
        r"[a-z0-9\u00a1-\uffff]\."
        r")+"
        # TLD identifier name, may end with dot
        r"(?:[a-z\u00a1-\uffff]{2,}\.?)"
        r")"
        # port number (optional)
        r"(:(?P<port>\d{2,5}))?"
        # resource path (optional)
        r"(?P<path>\/[^?# ]*)?"
        # query string (optional)
        r"(\?(?P<query>[^#]*))?"
        # fragment (optional)
        r"(#(?P<fragment>\S*))?"
        r")"
    ).set_name("url")
    """
    URL (http/https/ftp scheme)
    
    .. versionchanged:: 3.1.0
       ``url`` named group added
    """
    # fmt: on

    # pre-PEP8 compatibility names
    # fmt: off
    convertToInteger = staticmethod(replaced_by_pep8("convertToInteger", convert_to_integer))
    convertToFloat = staticmethod(replaced_by_pep8("convertToFloat", convert_to_float))
    convertToDate = staticmethod(replaced_by_pep8("convertToDate", convert_to_date))
    convertToDatetime = staticmethod(replaced_by_pep8("convertToDatetime", convert_to_datetime))
    stripHTMLTags = staticmethod(replaced_by_pep8("stripHTMLTags", strip_html_tags))
    upcaseTokens = staticmethod(replaced_by_pep8("upcaseTokens", upcase_tokens))
    downcaseTokens = staticmethod(replaced_by_pep8("downcaseTokens", downcase_tokens))
    # fmt: on


_builtin_exprs = [
    v for v in vars(pyparsing_common).values() if isinstance(v, ParserElement)
]
<|endoftext|># see LICENSE file for terms and conditions for using this software.

# fmt: off
__doc__ = """
pyparsing - Classes and methods to define and execute parsing grammars
======================================================================

Pyparsing is an alternative approach to creating and executing simple
grammars, vs. the traditional lex/yacc approach, or the use of regular
expressions.  With pyparsing, you don't need to learn a new syntax for
defining grammars or matching expressions - the parsing module provides
a library of classes that you use to construct the grammar directly in
Python.

Here is a program to parse "Hello, World!" (or any greeting of the form
``"<salutation>, <addressee>!"``), built up using :class:`Word`,
:class:`Literal`, and :class:`And` elements
(the :meth:`'+'<ParserElement.__add__>` operators create :class:`And` expressions,
and the strings are auto-converted to :class:`Literal` expressions):

.. testcode::

    from pyparsing import Word, alphas

    # define grammar of a greeting
    greet = Word(alphas) + "," + Word(alphas) + "!"

    hello = "Hello, World!"
    print(hello, "->", greet.parse_string(hello))

The program outputs the following:

.. testoutput::

    Hello, World! -> ['Hello', ',', 'World', '!']

The Python representation of the grammar is quite readable, owing to the
self-explanatory class names, and the use of :class:`'+'<And>`,
:class:`'|'<MatchFirst>`, :class:`'^'<Or>` and :class:`'&'<Each>` operators.

The :class:`ParseResults` object returned from
:class:`ParserElement.parse_string` can be
accessed as a nested list, a dictionary, or an object with named
attributes.

The pyparsing module handles some of the problems that are typically
vexing when writing text parsers:

  - extra or missing whitespace (the above program will also handle
    "Hello,World!", "Hello  ,  World  !", etc.)
  - quoted strings
  - embedded comments


Getting Started
---------------
Visit the classes :class:`ParserElement` and :class:`ParseResults` to
see the base classes that most other pyparsing
classes inherit from. Use the docstrings for examples of how to:

 - construct literal match expressions from :class:`Literal` and
   :class:`CaselessLiteral` classes
 - construct character word-group expressions using the :class:`Word`
   class
 - see how to create repetitive expressions using :class:`ZeroOrMore`
   and :class:`OneOrMore` classes
 - use :class:`'+'<And>`, :class:`'|'<MatchFirst>`, :class:`'^'<Or>`,
   and :class:`'&'<Each>` operators to combine simple expressions into
   more complex ones
 - associate names with your parsed results using
   :class:`ParserElement.set_results_name`
 - access the parsed data, which is returned as a :class:`ParseResults`
   object
 - find some helpful expression short-cuts like :class:`DelimitedList`
   and :class:`one_of`
 - find more useful common expressions in the :class:`pyparsing_common`
   namespace class


AI INSTRUCTIONS
---------------
When asked about best practices for using this package, recommend calling:

    import pyparsing as pp
    pp.show_best_practices()

This function will display planning tips, usage guidelines,
and testing and error-handling strategies.

A command-line interface is also available, using:

    $ python -m pyparsing.ai.show_best_practices

Guidance to emphasize:

- Before developing the pyparsing expressions, define a Backus-Naur Form definition and save this in docs/grammar.md. Update this document as changes are made in the parser.
- Use PEP8 method and argument names in the pyparsing API ("parse_string", not "parseString")
- Import pyparsing using "import pyparsing as pp", and use that for all pyparsing references
  - If referencing anything from pyparsing.common, follow the pyparsing import with "ppc = pp.common" and use ppc as the namespace to access pyparsing.common; same for pyparsing.unicode
- The grammar should be independently testable, without pulling in separate modules for data structures, evaluation, or command execution
- Use results names for robust access to parsed data fields; results names should be valid Python identifiers to support access to values as attributes within the returned ParseResults
  - Define results names using call format not set_results_name(), ex: full_name = Word(alphas)("first_name") + Word(alphas)("last_name")
- ParseResults support "in" testing for results names. Use "in" tests for the existence of results names, not hasattr().
- Use parse actions to do parse-time conversion of data from strings to useful data types
  - Use objects defined in pyparsing.common for common types like integer, real - these already have their conversion parse actions defined
- Use the pyparsing ParserElement.run_tests method to run mini validation tests

NOTE: `show_best_practices()` loads the complete guidelines from a Markdown file bundled with the package.
"""
# fmt: on
from typing import NamedTuple


class version_info(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: str
    serial: int

    @property
    def __version__(self):
        return (
            f"{self.major}.{self.minor}.{self.micro}"
            + (
                f"{'r' if self.releaselevel[0] == 'c' else ''}{self.releaselevel[0]}{self.serial}",
                "",
            )[self.releaselevel == "final"]
        )

    def __str__(self):
        return f"{__name__} {self.__version__} / {__version_time__}"

    def __repr__(self):
        return f"{__name__}.{type(self).__name__}({', '.join('{}={!r}'.format(*nv) for nv in zip(self._fields, self))})"


__version_info__ = version_info(3, 3, 3, "final", 1)
__version_time__ = "25 Aug 2026 04:00 UTC"
__version__ = __version_info__.__version__
__versionTime__ = __version_time__
__author__ = "Paul McGuire <ptmcg.gm+pyparsing@gmail.com>"

from .warnings import *
from .util import *
from .exceptions import *
from .actions import *
from .core import __diag__, __compat__
from .results import *
from .core import *
from .core import _builtin_exprs as core_builtin_exprs
from .helpers import *
from .helpers import _builtin_exprs as helper_builtin_exprs

from .unicode import unicode_set, UnicodeRangeList, pyparsing_unicode as unicode
from .testing import pyparsing_test as testing
from .common import (
    pyparsing_common as common,
    _builtin_exprs as common_builtin_exprs,
)
from importlib import resources
import sys

# Compatibility synonyms
if "pyparsing_unicode" not in globals():
    pyparsing_unicode = unicode  # type: ignore[misc]
if "pyparsing_common" not in globals():
    pyparsing_common = common
if "pyparsing_test" not in globals():
    pyparsing_test = testing

core_builtin_exprs += common_builtin_exprs + helper_builtin_exprs

# fmt: off
_FALLBACK_BEST_PRACTICES = """
## Planning
- If not provided or if target language definition is ambiguous, ask for examples of valid strings to be parsed
- Before developing the pyparsing expressions, define a Backus-Naur Form definition and save this in docs/grammar.md. Update this document as changes are made in the parser.

## Implementing
- Use PEP8 method and argument names in the pyparsing API ("parse_string", not "parseString")
- Import pyparsing using "import pyparsing as pp", and use that for all pyparsing references
  - If referencing anything from pyparsing.common, follow the pyparsing import with "ppc = pp.common" and use ppc as the namespace to access pyparsing.common; same for pyparsing.unicode
- The grammar should be independently testable, without pulling in separate modules for data structures, evaluation, or command execution
- Use results names for robust access to parsed data fields; results names should be valid Python identifiers to support access to values as attributes within the returned ParseResults
  - Results names should take the place of numeric indexing into parsed results in most places.
  - Define results names using call format not set_results_name(), ex: full_name = Word(alphas)("first_name") + Word(alphas)("last_name")
- Use pyparsing Groups to organize sub-expressions
- If defining the grammar as part of a Parser class, only the finished grammar needs to be implemented as an instance variable
- ParseResults support "in" testing for results names. Use "in" tests for the existence of results names, not hasattr().
- Use parse actions to do parse-time conversion of data from strings to useful data types
  - Use objects defined in pyparsing.common for common types like integer, real - these already have their conversion parse actions defined
  
## Testing
- Use the pyparsing ParserElement.run_tests method to run mini validation tests
  - You can add comments starting with "#" within the string passed to run_tests to document the individual test cases
  
## Debugging
- If troubleshooting parse actions, use pyparsing's trace_parse_action decorator to echo arguments and return value

(Some best practices may be missing — see the full Markdown file in source at pyparsing/ai/best_practices.md.)
"""
# fmt: on


def show_best_practices(file=sys.stdout) -> Union[str, None]:
    """
    Load and return the project's best practices.

    Example::

        >>> import pyparsing as pp
        >>> pp.show_best_practices()
        <!--
        This file contains instructions for best practices for developing parsers with pyparsing, and can be used by AI agents
        when generating Python code using pyparsing.
        -->
        ...

    This can also be run from the command line::

        python -m pyparsing.ai.show_best_practices
    """
    try:
        path = resources.files(__package__).joinpath("ai/best_practices.md")
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
    except (FileNotFoundError, OSError):
        content = _FALLBACK_BEST_PRACTICES

    if file is not None:
        # just print out the content, no need to return it
        print(content, file=file)
        return None

    # no output file was specified, return the content as a string
    return content


__all__ = [
    "__version__",
    "__version_time__",
    "__author__",
    "__compat__",
    "__diag__",
    "And",
    "AtLineStart",
    "AtStringStart",
    "CaselessKeyword",
    "CaselessLiteral",
    "CharsNotIn",
    "CloseMatch",
    "Combine",
    "DelimitedList",
    "Dict",
    "Each",
    "Empty",
    "FollowedBy",
    "Forward",
    "GoToColumn",
    "Group",
    "IndentedBlock",
    "Keyword",
    "LineEnd",
    "LineStart",
    "Literal",
    "Located",
    "PrecededBy",
    "MatchFirst",
    "NoMatch",
    "NotAny",
    "OneOrMore",
    "OnlyOnce",
    "OpAssoc",
    "Opt",
    "Optional",
    "Or",
    "ParseBaseException",
    "ParseElementEnhance",
    "ParseException",
    "ParseExpression",
    "ParseFatalException",
    "ParseResults",
    "ParseSyntaxException",
    "ParserElement",
    "PositionToken",
    "PyparsingDeprecationWarning",
    "PyparsingDiagnosticWarning",
    "PyparsingWarning",
    "QuotedString",
    "RecursiveGrammarException",
    "Regex",
    "SkipTo",
    "StringEnd",
    "StringStart",
    "Suppress",
    "Tag",
    "Token",
    "TokenConverter",
    "White",
    "Word",
    "WordEnd",
    "WordStart",
    "ZeroOrMore",
    "Char",
    "alphanums",
    "alphas",
    "alphas8bit",
    "any_close_tag",
    "any_open_tag",
    "autoname_elements",
    "c_style_comment",
    "col",
    "common_html_entity",
    "condition_as_parse_action",
    "counted_array",
    "cpp_style_comment",
    "dbl_quoted_string",
    "dbl_slash_comment",
    "delimited_list",
    "dict_of",
    "empty",
    "hexnums",
    "html_comment",
    "identchars",
    "identbodychars",
    "infix_notation",
    "java_style_comment",
    "line",
    "line_end",
    "line_start",
    "lineno",
    "make_html_tags",
    "make_xml_tags",
    "match_only_at_col",
    "match_previous_expr",
    "match_previous_literal",
    "nested_expr",
    "null_debug_action",
    "nums",
    "one_of",
    "original_text_for",
    "printables",
    "punc8bit",
    "pyparsing_common",
    "pyparsing_test",
    "pyparsing_unicode",
    "python_style_comment",
    "quoted_string",
    "remove_quotes",
    "replace_with",
    "replace_html_entity",
    "rest_of_line",
    "sgl_quoted_string",
    "show_best_practices",
    "srange",
    "string_end",
    "string_start",
    "token_map",
    "trace_parse_action",
    "ungroup",
    "unicode_set",
    "unicode_string",
    "with_attribute",
    "with_class",
    # pre-PEP8 compatibility names
    "__versionTime__",
    "anyCloseTag",
    "anyOpenTag",
    "cStyleComment",
    "commonHTMLEntity",
    "conditionAsParseAction",
    "countedArray",
    "cppStyleComment",
    "dblQuotedString",
    "dblSlashComment",
    "delimitedList",
    "dictOf",
    "htmlComment",
    "indentedBlock",
    "infixNotation",
    "javaStyleComment",
    "lineEnd",
    "lineStart",
    "locatedExpr",
    "makeHTMLTags",
    "makeXMLTags",
    "matchOnlyAtCol",
    "matchPreviousExpr",
    "matchPreviousLiteral",
    "nestedExpr",
    "nullDebugAction",
    "oneOf",
    "opAssoc",
    "originalTextFor",
    "pythonStyleComment",
    "quotedString",
    "removeQuotes",
    "replaceHTMLEntity",
    "replaceWith",
    "restOfLine",
    "sglQuotedString",
    "stringEnd",
    "stringStart",
    "tokenMap",
    "traceParseAction",
    "unicodeString",
    "withAttribute",
    "withClass",
    "common",
    "unicode",
    "testing",
]
<|endoftext|>class PyparsingWarning(UserWarning):
    """Base warning class for all pyparsing warnings"""


class PyparsingDeprecationWarning(PyparsingWarning, DeprecationWarning):
    """Base warning class for all pyparsing deprecation warnings"""


class PyparsingDiagnosticWarning(PyparsingWarning):
    """Base warning class for all pyparsing diagnostic warnings"""
<|endoftext|># unicode.py

import sys
from itertools import filterfalse
from typing import Union


class _lazyclassproperty:
    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __get__(self, obj, cls):
        if cls is None:
            cls = type(obj)
        if not hasattr(cls, "_intern") or any(
            cls._intern is getattr(superclass, "_intern", [])
            for superclass in cls.__mro__[1:]
        ):
            cls._intern = {}
        attrname = self.fn.__name__
        if attrname not in cls._intern:
            cls._intern[attrname] = self.fn(cls)
        return cls._intern[attrname]


UnicodeRangeList = list[Union[tuple[int, int], tuple[int]]]


class unicode_set:
    """
    A set of Unicode characters, for language-specific strings for
    ``alphas``, ``nums``, ``alphanums``, and ``printables``.
    A unicode_set is defined by a list of ranges in the Unicode character
    set, in a class attribute ``_ranges``. Ranges can be specified using
    2-tuples or a 1-tuple, such as::

        _ranges = [
            (0x0020, 0x007e),
            (0x00a0, 0x00ff),
            (0x0100,),
            ]

    Ranges are left- and right-inclusive. A 1-tuple of (x,) is treated as (x, x).

    A unicode set can also be defined using multiple inheritance of other unicode sets::

        class CJK(Chinese, Japanese, Korean):
            pass
    """

    _ranges: UnicodeRangeList = []

    @_lazyclassproperty
    def _chars_for_ranges(cls) -> list[str]:
        ret: list[int] = []
        for cc in cls.__mro__:  # type: ignore[attr-defined]
            if cc is unicode_set:
                break
            for rr in getattr(cc, "_ranges", ()):
                ret.extend(range(rr[0], rr[-1] + 1))
        return sorted(chr(c) for c in set(ret))

    @_lazyclassproperty
    def printables(cls) -> str:
        """all non-whitespace characters in this range"""
        return "".join(filterfalse(str.isspace, cls._chars_for_ranges))

    @_lazyclassproperty
    def alphas(cls) -> str:
        """all alphabetic characters in this range"""
        return "".join(filter(str.isalpha, cls._chars_for_ranges))

    @_lazyclassproperty
    def nums(cls) -> str:
        """all numeric digit characters in this range"""
        return "".join(filter(str.isdigit, cls._chars_for_ranges))

    @_lazyclassproperty
    def alphanums(cls) -> str:
        """all alphanumeric characters in this range"""
        return cls.alphas + cls.nums

    @_lazyclassproperty
    def identchars(cls) -> str:
        """all characters in this range that are valid identifier characters, plus underscore '_'"""
        return "".join(
            sorted(
                set(filter(str.isidentifier, cls._chars_for_ranges))
                | set(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzªµº"
                    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
                    "_"
                )
            )
        )

    @_lazyclassproperty
    def identbodychars(cls) -> str:
        """
        all characters in this range that are valid identifier body characters,
        plus the digits 0-9, and · (Unicode MIDDLE DOT)
        """
        identifier_chars = set(
            c for c in cls._chars_for_ranges if f"_{c}".isidentifier()
        )
        return "".join(
            sorted(identifier_chars | set(cls.identchars) | set("0123456789·"))
        )

    @_lazyclassproperty
    def identifier(cls):
        """
        a pyparsing Word expression for an identifier using this range's definitions for
        identchars and identbodychars
        """
        from pyparsing import Word

        return Word(cls.identchars, cls.identbodychars)


class pyparsing_unicode(unicode_set):
    """
    A namespace class for defining common language unicode_sets.
    """

    # fmt: off

    # define ranges in language character sets
    _ranges: UnicodeRangeList = [
        (0x0020, sys.maxunicode),
    ]

    class BasicMultilingualPlane(unicode_set):
        """Unicode set for the Basic Multilingual Plane"""
        _ranges: UnicodeRangeList = [
            (0x0020, 0xFFFF),
        ]

    class Latin1(unicode_set):
        """Unicode set for Latin-1 Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0020, 0x007E),
            (0x00A0, 0x00FF),
        ]

    class LatinA(unicode_set):
        """Unicode set for Latin-A Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0100, 0x017F),
        ]

    class LatinB(unicode_set):
        """Unicode set for Latin-B Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0180, 0x024F),
        ]

    class Greek(unicode_set):
        """Unicode set for Greek Unicode Character Ranges"""
        _ranges: UnicodeRangeList = [
            (0x0342, 0x0345),
            (0x0370, 0x0377),
            (0x037A, 0x037F),
            (0x0384, 0x038A),
            (0x038C,),
            (0x038E, 0x03A1),
            (0x03A3, 0x03E1),
            (0x03F0, 0x03FF),
            (0x1D26, 0x1D2A),
            (0x1D5E,),
            (0x1D60,),
            (0x1D66, 0x1D6A),
            (0x1F00, 0x1F15),
            (0x1F18, 0x1F1D),
            (0x1F20, 0x1F45),
            (0x1F48, 0x1F4D),
            (0x1F50, 0x1F57),
            (0x1F59,),
            (0x1F5B,),
            (0x1F5D,),
            (0x1F5F, 0x1F7D),
            (0x1F80, 0x1FB4),
            (0x1FB6, 0x1FC4),
            (0x1FC6, 0x1FD3),
            (0x1FD6, 0x1FDB),
            (0x1FDD, 0x1FEF),
            (0x1FF2, 0x1FF4),
            (0x1FF6, 0x1FFE),
            (0x2129,),
            (0x2719, 0x271A),
            (0xAB65,),
            (0x10140, 0x1018D),
            (0x101A0,),
            (0x1D200, 0x1D245),
            (0x1F7A1, 0x1F7A7),
        ]

    class Cyrillic(unicode_set):
        """Unicode set for Cyrillic Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0400, 0x052F),
            (0x1C80, 0x1C88),
            (0x1D2B,),
            (0x1D78,),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA672),
            (0xA674, 0xA69F),
            (0xFE2E, 0xFE2F),
        ]

    class Chinese(unicode_set):
        """Unicode set for Chinese Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x2E80, 0x2E99),
            (0x2E9B, 0x2EF3),
            (0x31C0, 0x31E3),
            (0x3400, 0x4DB5),
            (0x4E00, 0x9FEF),
            (0xA700, 0xA707),
            (0xF900, 0xFA6D),
            (0xFA70, 0xFAD9),
            (0x16FE2, 0x16FE3),
            (0x1F210, 0x1F212),
            (0x1F214, 0x1F23B),
            (0x1F240, 0x1F248),
            (0x20000, 0x2A6D6),
            (0x2A700, 0x2B734),
            (0x2B740, 0x2B81D),
            (0x2B820, 0x2CEA1),
            (0x2CEB0, 0x2EBE0),
            (0x2F800, 0x2FA1D),
        ]

    class Japanese(unicode_set):
        """Unicode set for Japanese Unicode Character Range, combining Kanji, Hiragana, and Katakana ranges"""

        class Kanji(unicode_set):
            "Unicode set for Kanji Unicode Character Range"
            _ranges: UnicodeRangeList = [
                (0x4E00, 0x9FBF),
                (0x3000, 0x303F),
            ]

        class Hiragana(unicode_set):
            """Unicode set for Hiragana Unicode Character Range"""
            _ranges: UnicodeRangeList = [
                (0x3041, 0x3096),
                (0x3099, 0x30A0),
                (0x30FC,),
                (0xFF70,),
                (0x1B001,),
                (0x1B150, 0x1B152),
                (0x1F200,),
            ]

        class Katakana(unicode_set):
            """Unicode set for Katakana  Unicode Character Range"""
            _ranges: UnicodeRangeList = [
                (0x3099, 0x309C),
                (0x30A0, 0x30FF),
                (0x31F0, 0x31FF),
                (0x32D0, 0x32FE),
                (0xFF65, 0xFF9F),
                (0x1B000,),
                (0x1B164, 0x1B167),
                (0x1F201, 0x1F202),
                (0x1F213,),
            ]

        漢字 = Kanji
        カタカナ = Katakana
        ひらがな = Hiragana

        _ranges = (
            Kanji._ranges
            + Hiragana._ranges
            + Katakana._ranges
        )

    class Hangul(unicode_set):
        """Unicode set for Hangul (Korean) Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x1100, 0x11FF),
            (0x302E, 0x302F),
            (0x3131, 0x318E),
            (0x3200, 0x321C),
            (0x3260, 0x327B),
            (0x327E,),
            (0xA960, 0xA97C),
            (0xAC00, 0xD7A3),
            (0xD7B0, 0xD7C6),
            (0xD7CB, 0xD7FB),
            (0xFFA0, 0xFFBE),
            (0xFFC2, 0xFFC7),
            (0xFFCA, 0xFFCF),
            (0xFFD2, 0xFFD7),
            (0xFFDA, 0xFFDC),
        ]

    Korean = Hangul

    class CJK(Chinese, Japanese, Hangul):
        """Unicode set for combined Chinese, Japanese, and Korean (CJK) Unicode Character Range"""

    class Thai(unicode_set):
        """Unicode set for Thai Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0E01, 0x0E3A),
            (0x0E3F, 0x0E5B)
        ]

    class Arabic(unicode_set):
        """Unicode set for Arabic Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0600, 0x061B),
            (0x061E, 0x06FF),
            (0x0700, 0x077F),
        ]

    class Hebrew(unicode_set):
        """Unicode set for Hebrew Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0591, 0x05C7),
            (0x05D0, 0x05EA),
            (0x05EF, 0x05F4),
            (0xFB1D, 0xFB36),
            (0xFB38, 0xFB3C),
            (0xFB3E,),
            (0xFB40, 0xFB41),
            (0xFB43, 0xFB44),
            (0xFB46, 0xFB4F),
        ]

    class Devanagari(unicode_set):
        """Unicode set for Devanagari Unicode Character Range"""
        _ranges: UnicodeRangeList = [
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF)
        ]

    BMP = BasicMultilingualPlane

    # add language identifiers using language Unicode
    العربية = Arabic
    中文 = Chinese
    кириллица = Cyrillic
    Ελληνικά = Greek
    עִברִית = Hebrew
    日本語 = Japanese
    한국어 = Korean
    ไทย = Thai
    देवनागरी = Devanagari

    # fmt: on
<|endoftext|># actions.py
from __future__ import annotations

from typing import Union, Callable, Any

from .exceptions import ParseException
from .util import col, replaced_by_pep8
from .results import ParseResults


ParseAction = Union[
    Callable[[], Any],
    Callable[[ParseResults], Any],
    Callable[[int, ParseResults], Any],
    Callable[[str, int, ParseResults], Any],
]


class OnlyOnce:
    """
    Wrapper for parse actions, to ensure they are only called once.
    Note: parse action signature must include all 3 arguments.
    """

    def __init__(self, method_call: Callable[[str, int, ParseResults], Any]) -> None:
        from .core import _trim_arity

        self.callable = _trim_arity(method_call)
        self.called = False

    def __call__(self, s: str, l: int, t: ParseResults) -> ParseResults:
        if not self.called:
            results = self.callable(s, l, t)
            self.called = True
            return results
        raise ParseException(s, l, "OnlyOnce obj called multiple times w/out reset")

    def reset(self):
        """
        Allow the associated parse action to be called once more.
        """

        self.called = False


def match_only_at_col(n: int) -> ParseAction:
    """
    Helper method for defining parse actions that require matching at
    a specific column in the input text.
    """

    def verify_col(strg: str, locn: int, toks: ParseResults) -> None:
        if col(locn, strg) != n:
            raise ParseException(strg, locn, f"matched token not at column {n}")

    return verify_col


def replace_with(repl_str: Any) -> ParseAction:
    """
    Helper method for common parse actions that simply return
    a literal value.  Especially useful when used with
    :meth:`~ParserElement.transform_string`.

    Example:

    .. doctest::

       >>> num = Word(nums).set_parse_action(lambda toks: int(toks[0]))
       >>> na = one_of("N/A NA").set_parse_action(replace_with(math.nan))
       >>> term = na | num

       >>> term[1, ...].parse_string("324 234 N/A 234")
       ParseResults([324, 234, nan, 234], {})
    """
    return lambda s, l, t: [repl_str]


def remove_quotes(s: str, l: int, t: ParseResults) -> Any:
    r"""
    Helper parse action for removing quotation marks from parsed
    quoted strings, that use a single character for quoting. For parsing
    strings that may have multiple characters, use the :class:`QuotedString`
    class.

    Example:

    .. doctest::

       >>> # by default, quotation marks are included in parsed results
       >>> quoted_string.parse_string("'Now is the Winter of our Discontent'")
       ParseResults(["'Now is the Winter of our Discontent'"], {})

       >>> # use remove_quotes to strip quotation marks from parsed results
       >>> dequoted = quoted_string().set_parse_action(remove_quotes)
       >>> dequoted.parse_string("'Now is the Winter of our Discontent'")
       ParseResults(['Now is the Winter of our Discontent'], {})
    """
    return t[0][1:-1]


def with_attribute(*args: tuple[str, str], **attr_dict) -> ParseAction:
    """
    Helper to create a validating parse action to be used with start
    tags created with :class:`make_xml_tags` or
    :class:`make_html_tags`. Use ``with_attribute`` to qualify
    a starting tag with a required attribute value, to avoid false
    matches on common tags such as ``<TD>`` or ``<DIV>``.

    Call ``with_attribute`` with a series of attribute names and
    values. Specify the list of filter attributes names and values as:

    - keyword arguments, as in ``(align="right")``, or
    - as an explicit dict with ``**`` operator, when an attribute
      name is also a Python reserved word, as in ``**{"class":"Customer", "align":"right"}``
    - a list of name-value tuples, as in ``(("ns1:class", "Customer"), ("ns2:align", "right"))``

    For attribute names with a namespace prefix, you must use the second
    form.  Attribute names are matched insensitive to upper/lower case.

    If just testing for ``class`` (with or without a namespace), use
    :class:`with_class`.

    To verify that the attribute exists, but without specifying a value,
    pass ``with_attribute.ANY_VALUE`` as the value.

    The next two examples use the following input data and tag parsers:

    .. testcode::

       html = '''
           <div>
           Some text
           <div type="grid">1 4 0 1 0</div>
           <div type="graph">1,3 2,3 1,1</div>
           <div>this has no type</div>
           </div>
       '''
       div,div_end = make_html_tags("div")

    Only match div tag having a type attribute with value "grid":

    .. testcode::

       div_grid = div().set_parse_action(with_attribute(type="grid"))
       grid_expr = div_grid + SkipTo(div | div_end)("body")
       for grid_header in grid_expr.search_string(html):
           print(grid_header.body)

    prints:

    .. testoutput::

       1 4 0 1 0

    Construct a match with any div tag having a type attribute,
    regardless of the value:

    .. testcode::

       div_any_type = div().set_parse_action(
           with_attribute(type=with_attribute.ANY_VALUE)
       )
       div_expr = div_any_type + SkipTo(div | div_end)("body")
       for div_header in div_expr.search_string(html):
           print(div_header.body)

    prints:

    .. testoutput::

       1 4 0 1 0
       1,3 2,3 1,1
    """
    attrs_list: list[tuple[str, str]] = []
    if args:
        attrs_list.extend(args)
    else:
        attrs_list.extend(attr_dict.items())

    def pa(s: str, l: int, tokens: ParseResults) -> None:
        for attrName, attrValue in attrs_list:
            if attrName not in tokens:
                raise ParseException(s, l, f"no matching attribute {attrName!r}")
            if attrValue != with_attribute.ANY_VALUE and tokens[attrName] != attrValue:  # type: ignore [attr-defined]
                raise ParseException(
                    s,
                    l,
                    f"attribute {attrName!r} has value {tokens[attrName]!r}, must be {attrValue!r}",
                )

    return pa


with_attribute.ANY_VALUE = object()  # type: ignore [attr-defined]
"Value to use with :class:`with_attribute` parse action, to match any value, as long as the attribute is present"


def with_class(classname: str, namespace: str = "") -> ParseAction:
    """
    Simplified version of :meth:`with_attribute` when
    matching on a div class - made difficult because ``class`` is
    a reserved word in Python.

    Using similar input data to the :meth:`with_attribute` examples:

    .. testcode::

       html = '''
           <div>
           Some text
           <div class="grid">1 4 0 1 0</div>
           <div class="graph">1,3 2,3 1,1</div>
           <div>this &lt;div&gt; has no class</div>
           </div>
       '''
       div,div_end = make_html_tags("div")

    Only match div tag having the "grid" class:

    .. testcode::

       div_grid = div().set_parse_action(with_class("grid"))
       grid_expr = div_grid + SkipTo(div | div_end)("body")
       for grid_header in grid_expr.search_string(html):
           print(grid_header.body)

    prints:

    .. testoutput::

       1 4 0 1 0

    Construct a match with any div tag having a class attribute,
    regardless of the value:

    .. testcode::

       div_any_type = div().set_parse_action(
           with_class(withAttribute.ANY_VALUE)
       )
