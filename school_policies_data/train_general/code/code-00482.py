#: The characters at the end of an over-long name that carry a digest of the part cut off.
_HOST_NAME_DIGEST_LEN: Final[int] = 8
#: The kernel ABI fixes the initial PID namespace's inode (``PID_NS_INIT_INO`` in ``linux/nsfs.h``); every namespace
#: created later draws one from ``0xF0000000`` up, so this value names the host's own namespace.
_INITIAL_PID_NAMESPACE: Final[int] = 0xEFFFFFFC
_NAMESPACE_MARK: Final[str] = "?pidns-"


def host_name() -> str:
    """
    The hostname recorded alongside an owner, so a marker written on another machine is never probed here.

    Marker formats here hold the field to printable non-space ASCII, while ``socket.gethostname()`` reports whatever
    the kernel stores: a space, a newline that forges an extra marker line, or a byte no codec encodes, which Python
    hands back as a surrogate. A holder that publishes one raw writes a marker it cannot read back, and loses the lock
    to the first peer that ages it out as malformed. ``?`` is illegal in a hostname, so escaping it along with every
    out-of-grammar byte as ``?<hex>`` leaves a real name untouched and still keeps two hosts apart, which
    :func:`owner_is_stale` relies on to refuse to probe a foreign PID.

    Linux appends the PID namespace, as Mercurial's lock prefix does. Two containers in one pod share the hostname and
    the boot id but not the PID namespace, so a PID recorded in one names another process, or none, in the other. The
    suffix starts with a ``?`` no escape can produce, so ``a`` in a namespace never reads as a host literally named
    ``a?pidns-...``. The initial namespace adds nothing, which keeps a host's name what older releases recorded.

    An over-long name ends in a digest of the part cut to fit the limit. A bare cut gives every name that shares the
    kept prefix one identity, and :func:`owner_is_stale` would then probe a local PID for another host's holder.
    """
    name = (
        "".join(
            chr(byte) if byte in _HOST_NAME_VERBATIM else f"?{byte:02x}"
            for byte in socket.gethostname().encode("utf-8", "surrogateescape")
        )
        + _pid_namespace_suffix()
    )
    if len(name) > _HOST_NAME_LIMIT:
        keep = _HOST_NAME_LIMIT - _HOST_NAME_DIGEST_LEN - 1
        name = f"{name[:keep]}-{hashlib.sha256(name[keep:].encode()).hexdigest()[:_HOST_NAME_DIGEST_LEN]}"
    # An escape is three characters and a kept byte is never '?', so a bare '?' can only mean an empty hostname.
    return name or "?"


def _pid_namespace_suffix() -> str:
    try:
        namespace: Final = Path("/proc/self/ns/pid").stat().st_ino
    except OSError:  # only Linux 3.8+ has the file, and a sandbox may hide it; nothing else tells namespaces apart
        return ""
    return "" if namespace == _INITIAL_PID_NAMESPACE else f"{_NAMESPACE_MARK}{namespace:x}"


def owner_is_stale(pid: int, hostname: str, start_token: int | None) -> bool:
    """
    Whether the recorded owner is provably gone, so reclaiming its marker cannot detach a live holder.

    Fail closed: return ``True`` only when this process can prove the exact recorded owner is dead. A marker from
    another host cannot be probed; a live PID whose start token still matches, or whose token cannot be read, is the
    holder or is indistinguishable from it; a live PID whose start token differs is a recycled PID, so the process that
    wrote the marker is gone. PostgreSQL, Qt ``QLockFile`` and Mercurial all break a stale lock only on proof of death
    and treat an unreadable or foreign owner as still holding.
    """
    if hostname != host_name():
        return False
    if process_alive(pid) and (start_token is None or process_start_token(pid) in {None, start_token}):
        return False
    # Only a bare hostname can be an older release's, written from a container that shares this host's name.
    return _NAMESPACE_MARK in hostname or not _owner_runs_elsewhere(pid, start_token)


def owner_is_current_process(pid: int, hostname: str, start_token: int | None) -> bool:
    """
    Whether the recorded owner is this process, so a marker a recycled PID left behind never reads as ours.

    Fail closed: a recorded start token must match this process's. One that differs was written by an earlier process
    that reused this PID, and when this process cannot read its own token nothing proves the marker is ours. A marker
    without a token falls back to the PID and hostname.
    """
    if pid != os.getpid() or hostname != host_name():
        return False
    return start_token is None or start_token == process_start_token(pid)


if sys.platform == "win32":  # pragma: win32 cover
    import ctypes
    from ctypes import wintypes

    _KERNEL32: Final[ctypes.WinDLL] = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    _KERNEL32.GetProcessTimes.restype = wintypes.BOOL

    _WIN_SYNCHRONIZE: Final[int] = 0x100000
    _WIN_PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000
    _WIN_ERROR_INVALID_PARAMETER: Final[int] = 87
    _WIN_INHERIT_HANDLE: Final[bool] = False

    def process_alive(pid: int) -> bool:
        """Whether a process with this PID exists, treating an access denial as proof it does."""
        handle = _KERNEL32.OpenProcess(_WIN_SYNCHRONIZE, _WIN_INHERIT_HANDLE, pid)
        if handle:
            _KERNEL32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() != _WIN_ERROR_INVALID_PARAMETER

    def process_start_token(pid: int) -> int | None:
        """The process creation FILETIME as a 100ns tick count, or ``None`` when it cannot be read."""
        handle = _KERNEL32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, _WIN_INHERIT_HANDLE, pid)
        if not handle:
            return None
        creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        try:
            if not _KERNEL32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return None  # pragma: no cover  # win32 GetProcessTimes failure path; not reproducible on a live handle
        finally:
            _KERNEL32.CloseHandle(handle)
        return (creation.dwHighDateTime << 32) | creation.dwLowDateTime

else:  # pragma: win32 no cover

    def process_alive(pid: int) -> bool:
        """Whether a process with this PID exists, treating an access denial (``EPERM``) as proof it does."""
        try:
            os.kill(pid, 0)
        except OSError as error:
            if error.errno == ESRCH:
                return False
            if error.errno == EPERM:
                return True
            raise
        return True

    if sys.platform in {"linux", "android"}:  # pragma: linux cover
        # Termux/Android reports sys.platform == "android" but runs the Linux kernel, so /proc/<pid>/stat and the boot
        # id are the same reliable start-time source; treat it exactly like Linux rather than the tokenless fallback.
        # comm (field 2) is wrapped in parentheses and may itself contain spaces or a ')', so the fixed fields start
        # after the final ')'. starttime is field 22 overall, the twentieth of those trailing fields (index 19).
        _STARTTIME_INDEX: Final[int] = 19

        def _read_boot_id() -> int:
            # starttime is measured in clock ticks since boot, so on its own it repeats across a reboot. Folding the
            # boot id into the high bits makes the Linux token reboot-safe like the absolute clocks macOS and Windows
            # expose, while staying a single integer so a 3.29 reader still parses the third marker line. 0 when the
            # kernel does not expose a boot id degrades to bare starttime, which stays safe (a reboot collision fails
            # closed rather than reclaiming a live marker).
            try:
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii")
                return int(boot_id.strip().replace("-", ""), 16)
            except (OSError, ValueError):  # pragma: no cover  # the kernel always exposes boot_id as a UUID on Linux
                return 0

        _BOOT_ID: Final[int] = _read_boot_id()

        def process_start_token(pid: int) -> int | None:
            """The ``/proc/<pid>/stat`` ``starttime`` folded with the boot id, or ``None`` when the process is gone."""
            try:
                data = Path(f"/proc/{pid}/stat").read_bytes()
            except OSError:
                return None
            # psutil identifies a process by the same (pid, starttime) pair; the boot id extends that across reboots.
            fields = data[data.rfind(b")") + 1 :].split()
            if len(fields) <= _STARTTIME_INDEX:  # pragma: no cover  # a truncated /proc read, never seen in practice
                return None
            try:
                starttime = int(fields[_STARTTIME_INDEX])
            except ValueError:  # pragma: no cover  # /proc always renders starttime as an integer
                return None
            return (_BOOT_ID << 64) | starttime

        def _owner_runs_elsewhere(pid: int, start_token: int | None) -> bool:
            """
            Whether the owner runs in another PID namespace as *pid*.

            An older release in a container sharing this hostname recorded its in-container PID, which means nothing
            here. starttime and the boot id are kernel-wide, so a task with that start token whose innermost PID is
            *pid* is the owner; one that merely started in the same tick is not.
            """
            if start_token is None or start_token >> 64 != _BOOT_ID:
                return False
            try:
                with os.scandir("/proc") as entries:
                    return any(
                        entry.name.isdigit()
                        and (task := int(entry.name)) != pid
                        and process_start_token(task) == start_token
                        and _innermost_pid(task) == pid
                        for entry in entries
                    )
            except OSError:
                # A /proc this process cannot list proves nothing either way, which leaves 4.0.12's verdict.
                return False

        def _innermost_pid(task: int) -> int | None:
            try:
                status: Final = Path(f"/proc/{task}/status").read_text(encoding="ascii", errors="replace")
            except OSError:  # pragma: no cover  # the task exited between the stat read and this one
                return None
            return next(
                (int(line.split()[-1]) for line in status.splitlines() if line.startswith("NSpid:")),
                None,  # kernels before 4.1 record no namespace PIDs
            )

    elif sys.platform == "darwin":  # pragma: darwin cover
        import ctypes
        import struct

        _LIBC: Final[ctypes.CDLL] = ctypes.CDLL(None, use_errno=True)
        _CTL_KERN: Final[int] = 1
        _KERN_PROC: Final[int] = 14
        _KERN_PROC_PID: Final[int] = 1
        # kinfo_proc opens with kp_proc.p_starttime (a struct timeval) at offset 0: int64 seconds, int32 microseconds.
        # The offset is fixed by the struct chain kinfo_proc -> extern_proc -> p_un, so a read at 0 is not a guess.
        _TIMEVAL_AT_ZERO: Final[str] = "<qi"
        _TIMEVAL_SIZE: Final[int] = struct.calcsize(_TIMEVAL_AT_ZERO)

        def process_start_token(pid: int) -> int | None:
            """The process start time in microseconds from ``sysctl(KERN_PROC_PID)``, or ``None`` when it is gone."""
            mib = (ctypes.c_int * 4)(_CTL_KERN, _KERN_PROC, _KERN_PROC_PID, pid)
            length = ctypes.c_size_t(0)
            # The size probe reports the kinfo_proc size for any PID, so the read below, not the probe, tells a live
            # process from a gone one: a gone PID leaves the fetch a zero-length success, so re-check the length after.
            if _LIBC.sysctl(mib, 4, None, ctypes.byref(length), None, 0) != 0:
                return None
            buffer = (ctypes.c_char * length.value)()
            if _LIBC.sysctl(mib, 4, buffer, ctypes.byref(length), None, 0) != 0 or length.value < _TIMEVAL_SIZE:
                return None
            seconds, microseconds = struct.unpack_from(_TIMEVAL_AT_ZERO, buffer.raw, 0)
            return seconds * 1_000_000 + microseconds

    else:  # pragma: no cover  # a POSIX platform without a proven start-time source falls back to fail-closed liveness

        def process_start_token(pid: int) -> int | None:
            """No proven start-time source, so the owner carries no token and liveness rests on the PID alone."""
            del pid
            return None


if sys.platform not in {"linux", "android"}:  # pragma: linux no cover  # pragma: android no cover

    def _owner_runs_elsewhere(pid: int, start_token: int | None) -> bool:
        """Without Linux PID namespaces, a PID that is gone or recycled leaves its owner nowhere else to run."""
        del pid, start_token
        return False


__all__ = [
    "host_name",
    "owner_is_current_process",
    "owner_is_stale",
    "process_alive",
    "process_start_token",
]
<|endoftext|>from __future__ import annotations

import math
import os
import re
from contextlib import suppress
from typing import Final, Literal, NamedTuple

from ._identity import host_name, owner_is_current_process, owner_is_stale, process_start_token
from ._soft import SoftFileLock, _read_lock_file, parse_decimal
from ._util import break_lock_file, write_all

#: Protocol 1 is the legacy ``<pid>\n<hostname>\n[<start_token>\n]`` marker that :class:`SoftFileLock` still writes.
#: Protocol 2 carries the owner mode and the lease claim. A protocol 1 reader treats a protocol 2 marker as malformed
#: and evicts it after its grace period, so the two never guarantee mutual exclusion against each other.
_PROTOCOL: Final[str] = "filelock/2"

_MAX_PID: Final[int] = 2**31 - 1

#: The forms ``repr`` gives a non-negative finite float or int; ``float()`` would also accept whitespace, underscores
#: and non-ASCII digits.
_DURATION: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:e[+-][0-9]+)?", re.ASCII)

#: Preserve unknown contracts so contenders cannot mistake them for malformed, reclaimable markers.
OwnerMode = Literal["lease", "exclusive", "unknown"]


class MarkerSoftFileLock(SoftFileLock):
    """An existence lock whose marker carries a protocol 2 owner record."""

    #: Age-based expiry requires a lease contract; an exclusive holder grants none.
    _owner_mode: OwnerMode = "exclusive"

    @property
    def owner(self) -> OwnerRecord | None:
        """
        The owner named by the marker on disk.

        :returns: the published record, or ``None`` when no marker exists or its record is malformed or protocol 1

        """
        return self._read_owner()

    @property
    def pid(self) -> int | None:
        """
        The PID of the process holding this lock, read from the marker.

        :returns: the PID, or ``None`` when no marker exists or its record is unreadable

        """
        return None if (owner := self._read_owner()) is None else owner.pid

    @property
    def is_lock_held_by_us(self) -> bool:
        """
        Whether the marker on disk names this process.

        :returns: ``True`` when the marker names this process's PID, hostname and, when recorded, start token

        """
        owner = self._read_owner()
        return owner is not None and owner_is_current_process(owner.pid, owner.hostname, owner.start)

    def force_break(self) -> None:
        """
        Remove the marker whoever holds it, so a later contender can acquire.

        Forced breaking voids mutual exclusion: the previous holder keeps running and keeps using whatever the lock
        protects. Reserve it for an operator clearing a marker whose holder is known to be gone.
        """
        self.break_lock()

    def _try_break_stale_lock(self) -> None:
        with suppress(OSError, ValueError):
            snapshot: Final[tuple[str | None, float, int]] = _read_lock_file(self.lock_file)
            owner: Final[OwnerRecord | None]
            if (owner := parse_marker(snapshot[0])) is None:
                super()._try_break_stale_lock()
            elif owner.mode != "unknown" and owner_is_stale(owner.pid, owner.hostname, owner.start):
                break_lock_file(self.lock_file, *snapshot[1:])

    @classmethod
    def _recorded_holder(cls, content: str | None) -> tuple[int, str] | None:
        if (owner := parse_marker(content)) is not None:
            return owner.pid, owner.hostname
        return super()._recorded_holder(content)

    def _read_owner(self) -> OwnerRecord | None:
        with suppress(OSError, ValueError):
            return parse_marker(_read_lock_file(self.lock_file)[0])
        return None

    def _write_lock_info(self, fd: int) -> None:
        write_all(fd, encode_marker(self._published_record()))

    def _published_record(self) -> OwnerRecord:
        return OwnerRecord(
            pid=os.getpid(),
            hostname=host_name(),
            mode=self._owner_mode,
            start=process_start_token(os.getpid()),
        )


class OwnerRecord(NamedTuple):
    """Keep optional metadata when reading unknown lock modes."""

    pid: int
    hostname: str
    mode: OwnerMode
    token: str | None = None
    lease_duration: float | None = None
    start: int | None = None


def encode_marker(record: OwnerRecord) -> bytes:
    """Render an owner record as the bytes a protocol 2 marker holds."""
    lines = [_PROTOCOL, f"pid={record.pid}", f"host={record.hostname}", f"mode={record.mode}"]
    if record.token is not None:
        lines.append(f"token={record.token}")
    if record.lease_duration is not None:
        lines.append(f"duration={record.lease_duration!r}")
    if record.start is not None:
        lines.append(f"start={record.start}")
    return "".join(f"{line}\n" for line in lines).encode()


def parse_marker(content: str | None) -> OwnerRecord | None:
    """Return the owner a protocol 2 marker names, or ``None`` when the record is malformed or protocol 1."""
    # A torn write may end in a prefix like "mode=exclu", an unknown contract nobody reclaims; malformed, it ages out.
    if not content or content[-1] != "\n" or not (lines := content.strip().splitlines()) or lines[0] != _PROTOCOL:
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator:
            return None
        fields[key] = value
    return _build_record(fields)


def _build_record(fields: dict[str, str]) -> OwnerRecord | None:
    # An unknown key is a field a newer filelock published, so ignore it rather than read the record as malformed. An
    # unrecognized mode is the same story one level up: a contract this version does not implement. Reading it as
    # malformed would age the marker out of a live owner's hands, so keep it and let the caller refuse to reclaim it.
    # A record naming no mode at all states no contract and stays malformed.
    if (published := fields.get("mode")) is None:
        return None
    mode: Final[OwnerMode] = published if published in {"lease", "exclusive"} else "unknown"
    hostname = fields.get("host")
    if not hostname or "pid" not in fields or ("duration" in fields and not _DURATION.fullmatch(fields["duration"])):
        return None
    try:
        pid = parse_decimal(fields["pid"])
        duration = float(fields["duration"]) if "duration" in fields else None
        start = parse_decimal(fields["start"]) if "start" in fields else None
    except ValueError:
        return None
    if not 1 <= pid <= _MAX_PID:
        return None
    token = fields.get("token")
    # "1e400" passes the grammar but overflows to inf, which mismatches every configured duration and so wedges reclaim,
    # where a malformed marker ages out.
    if mode == "lease" and (not token or duration is None or not (math.isfinite(duration) and duration > 0)):
        return None
    return OwnerRecord(pid=pid, hostname=hostname, mode=mode, token=token, lease_duration=duration, start=start)


__all__ = [
    "MarkerSoftFileLock",
    "OwnerMode",
    "OwnerRecord",
    "encode_marker",
    "parse_marker",
]
<|endoftext|>from __future__ import annotations

import os
import secrets
import stat
import sys
from contextlib import suppress
from errno import EACCES, EIO, EISDIR
from pathlib import Path
from typing import Final


def write_all(fd: int, data: bytes) -> None:
    """
    Write the whole buffer to *fd*, looping over the short writes POSIX lets ``os.write`` make.

    A caller that stopped after one short write would leave a truncated marker behind. A peer that reads between two
    writes sees a prefix of the record; each caller has to tolerate that. Lock records need not survive a power loss, so
    we skip ``fsync``.

    :param fd: file descriptor open for writing.
    :param data: bytes to write in full.

    :raises OSError: if a write reports zero progress before the record is complete.

    """
    remaining = memoryview(data)
    while remaining:
        if (written := os.write(fd, remaining)) == 0:
            raise OSError(EIO, "os.write wrote 0 bytes before the record was complete")
        remaining = remaining[written:]


def raise_on_not_writable_file(filename: str) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.

    Separates files that can never be written from files that are writable but currently locked.

    :param filename: file to check

    :raises OSError: as if the file was opened for writing.

    """
    try:
        # lstat, not stat: settles exists-and-writable in one syscall, and a hostile symlink at the lock path would
        # make stat inspect the link target, letting an attacker turn a contended acquire into a misleading
        # PermissionError / IsADirectoryError and probe that target's attributes. The real open passes O_NOFOLLOW and
        # refuses the symlink anyway.
        file_stat = os.lstat(filename)
    except OSError:
        return  # does not exist, or an error the caller cannot act on

    # No mtime guard: the old `if st_mtime != 0` skip covered NFS/Linux quirks where os.lstat returned an all-zero
    # struct, which it no longer does. Skipping on mtime 0 let a read-only file or a directory at the lock path pass
    # as missing, so acquire() blocked forever on an open that cannot succeed.
    # Match open() credentials where supported; group permissions and ACLs can grant access without S_IWUSR.
    if not (file_stat.st_mode & stat.S_IWUSR) and not os.access(
        filename, os.W_OK, effective_ids=os.access in os.supports_effective_ids
    ):
        try:
            os.lstat(filename)
        except FileNotFoundError:
            # A holder may unlink its marker between lstat() and access(); let the caller attempt creation.
            return
        raise PermissionError(EACCES, "Permission denied", filename)

    if stat.S_ISDIR(file_stat.st_mode):
        if sys.platform == "win32":  # pragma: win32 cover
            raise PermissionError(EACCES, "Permission denied", filename)
        raise IsADirectoryError(EISDIR, "Is a directory", filename)  # pragma: win32 no cover


def ensure_directory_exists(filename: Path | str) -> None:
    """
    Ensure the directory containing the file exists (create it if necessary).

    :param filename: file.

    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def break_lock_file(lock_file: str, mtime_before: float, ino_before: int) -> None:
    """
    Remove the lock file a caller found stale with modification time *mtime_before* and inode *ino_before*.

    Right before renaming we ``lstat`` the lock again and leave it alone if a peer recreated it since the caller judged
    it stale, as Mercurial re-reads a lock under its break lock. We rename before unlinking so that, of several
    processes racing to break the same lock, one takes the file and the rest get ``OSError``. A newer modification time
    or another inode on the renamed file means a peer recreated the lock between that ``lstat`` and the rename, so we
    hard-link it back and drop the break name. ``link`` never replaces a name, so a marker a third process created after
    our rename wins and the live one stays under the break name. In the two syscalls between rename and link a third
    process can acquire beside the live holder, a soft lock by creating its marker or a StrictSoftFileLock by
    publishing its sentinel at the empty path. Without ``os.link``, or on a filesystem that refuses hard links, the live
    marker stays under the break name. We compare inodes as well as modification times because NFS and FAT store
    coarse modification times, and a peer that recreates the lock within that granularity leaves the old mtime on it.
    We call ``lstat`` to avoid following a symlink a peer swaps in after our stale check, and put back only a regular
    file because ``link`` would follow a symlink to its target.

    We add a random token to the break name so other processes cannot guess it and two breakers in one process do not
    share ``<lock>.break.<pid>``. With a shared name, the second breaker could rename a recreated live lock onto that
    path between our ``lstat`` and ``unlink``, and our ``unlink`` would delete it.

    :param lock_file: path to the lock file to break.
    :param mtime_before: modification time the caller saw when it judged the lock stale.
    :param ino_before: inode number the caller saw when it judged the lock stale.

    :raises OSError: if the rename, the re-check or the restore fails, for example because the file vanished or another
        user owns it in a sticky directory.

    """
    if (current := os.lstat(lock_file)).st_mtime > mtime_before or current.st_ino != ino_before:
        return
    break_path: Final[str] = f"{lock_file}.break.{os.getpid()}.{secrets.token_hex(16)}"
    Path(lock_file).rename(break_path)
    if (st_after := os.lstat(break_path)).st_mtime <= mtime_before and st_after.st_ino == ino_before:
        Path(break_path).unlink()
    elif stat.S_ISREG(st_after.st_mode) and _HAS_LINK:  # pragma: needs hard-link
        with suppress(FileExistsError):
            os.link(break_path, lock_file)
            Path(break_path).unlink()


def touch(name: str, *, fd: int) -> None:
    # Prefer the already-open, already-verified fd so a peer that swaps a symlink or a different file in at the
    # path after our O_NOFOLLOW read cannot redirect the touch: utime then targets the inode behind the fd.
    # Where the platform cannot utime an fd, fall back to a path-based touch that still refuses to follow a
    # symlink where supported, matching the O_NOFOLLOW reads used elsewhere here.
    if _SUPPORTS_UTIME_FD:  # pragma: needs utime-fd
        os.utime(fd, None)
        return
    os.utime(name, None, follow_symlinks=not _SUPPORTS_UTIME_NOFOLLOW)  # pragma: lacks utime-fd


# Termux/Android CPython ships without os.link.
_HAS_LINK: Final[bool] = hasattr(os, "link")
# Retargeting os.utime to an open fd lets a heartbeat refresh the exact inode it verified instead of whatever the
# pathname now names.
_SUPPORTS_UTIME_FD: Final[bool] = sys.platform != "win32" and os.utime in os.supports_fd
# os.utime follows symlinks unless told not to; not every platform can refuse the follow, so probe support.
_SUPPORTS_UTIME_NOFOLLOW: Final[bool] = os.utime in os.supports_follow_symlinks


__all__ = [
    "break_lock_file",
    "ensure_directory_exists",
    "raise_on_not_writable_file",
    "touch",
    "write_all",
]
<|endoftext|>from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
import secrets
import stat
import sys
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from dataclasses import dataclass, fields
from decimal import Decimal
from itertools import count, pairwise, starmap
from numbers import Real
from threading import TIMEOUT_MAX, Condition, RLock, Thread, current_thread, get_ident, local
from typing import TYPE_CHECKING, Final, Generic, Literal, NoReturn, TypedDict, TypeVar, cast
from weakref import WeakKeyDictionary, WeakValueDictionary

from ._error import SoftFileLockLifetimeWarning, Timeout
from ._util import break_lock_file

#: No explicit file permission mode was passed. Lock files then open with 0o666 so umask and default ACLs pick
#: the final permissions, and fchmod is skipped to preserve POSIX default ACL inheritance.
_UNSET_FILE_MODE: Final[int] = -1

_PERMISSION_BITS: Final[int] = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO

#: Ceiling on the retry counter used as a power of two, so a long contended wait cannot overflow the backoff multiply.
_MAX_BACKOFF_EXPONENT: Final[int] = 20

#: How a context manager reconciles a body failure with a release failure on exit (see the property of this name).
ContextErrorPolicy = Literal["chain", "group"]
_CONTEXT_ERROR_POLICIES: Final[frozenset[str]] = frozenset({"chain", "group"})

#: What a descriptor-owning backend does with an ``os.close`` failure after relinquishing ownership (see the property).
CloseErrorPolicy = Literal["default", "raise", "suppress"]
_CLOSE_ERROR_POLICIES: Final[frozenset[str]] = frozenset({"default", "raise", "suppress"})

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from types import FrameType, TracebackType
    from typing import Protocol

    from _typeshed import Unused

    from ._lease import _LeaseClaim
    from ._read_write import ReadWriteLock
    from ._soft_rw import SoftReadWriteLock

    class _ForkResettable(Protocol):
        def _reset_after_fork_in_child(self) -> None: ...

    class _ForkDescriptorOwner(Protocol):
        def _descriptors_for_fork(self) -> tuple[tuple[int, tuple[int, int] | None], ...]: ...

    # Matched against the class object itself rather than `type[...]` of it. A metaclass supplies this method to the
    # class while leaving instances without it, so a `type[_ForkResettableClass]` bound rejects `ReadWriteLock`.
    class _ForkResettableClass(Protocol):
        def _reset_class_after_fork(self) -> None: ...

    class _RegisterAtFork(Protocol):
        def __call__(
            self,
            *,
            before: Callable[[], None] | None = None,
            after_in_parent: Callable[[], None] | None = None,
            after_in_child: Callable[[], None] | None = None,
        ) -> None: ...

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Self
    else:  # pragma: no cover (<py311)
        from typing_extensions import Self

_LOGGER: Final[logging.Logger] = logging.getLogger("filelock")
_REGISTER_AT_FORK: Final[_RegisterAtFork | None] = cast("_RegisterAtFork | None", getattr(os, "register_at_fork", None))
_HAS_REGISTER_AT_FORK: Final[bool] = _REGISTER_AT_FORK is not None

_ExtraValue = TypeVar("_ExtraValue")
_MarkerValue = TypeVar("_MarkerValue")
_SubclassValue = TypeVar("_SubclassValue")
_LockInitValue = float | int | bool | str | None | Callable[[int], None]


class LockOptions(TypedDict, total=False):
    """Every option the metaclass forwards, so a subclass adding its own can still type what it passes through."""

    timeout: float
    mode: int
    thread_local: bool
    blocking: bool
    is_singleton: bool
    poll_interval: float
    lifetime: float | None
    context_error_policy: ContextErrorPolicy
    close_error_policy: CloseErrorPolicy
    fallback_to_soft: bool
    preserve_lock_file: bool
    on_acquired: Callable[[int], None] | None


def _exception_group_cls() -> type[BaseException] | None:
    # filelock has no runtime dependencies, so the 3.10 exceptiongroup backport is optional: without it grouped errors
    # chain instead, and only context_error_policy="group" is refused.
    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        return BaseExceptionGroup  # ruff:ignore[undefined-name]  # builtin on 3.11+
    try:  # pragma: <3.11 cover
        # Alias the import so BaseExceptionGroup above stays the builtin rather than an unbound local of this function.
        from exceptiongroup import (  # ruff:ignore[import-outside-top-level]  # optional backport
            BaseExceptionGroup as _Backport,
        )
    except ImportError:  # pragma: <3.11 cover
        return None
    return _Backport  # pragma: <3.11 cover


def _is_exception_group(error: BaseException) -> bool:
    return (group_cls := _exception_group_cls()) is not None and isinstance(error, group_cls)


def _raise_grouped_errors(
    message: str,
    first_error: BaseException,
    second_error: BaseException,
    *additional_errors: BaseException,
    marker: tuple[str, _MarkerValue] | None = None,
) -> NoReturn:
    errors = (first_error, second_error, *additional_errors)
    if (group_cls := _exception_group_cls()) is None:  # pragma: <3.11 cover
        # Chain them the way nested handlers would: the last error propagates with the earlier ones as its context.
        for earlier, later in pairwise(errors):
            _append_exception_context(later, earlier)
        _raise_chained_errors(errors[-1])
    _detach_grouped_contexts(errors)
    group: Final = group_cls(message, errors)
    if marker is not None:
        setattr(group, marker[0], marker[1])
    raise group from None


def _detach_grouped_contexts(errors: tuple[BaseException, ...]) -> None:
    seen: set[int] = set()
    pending = list(errors)
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        if (context := error.__context__) is not None and (
            context is error
            or _same_exception_tree(error, context)
            or any(context is root or _contains_exception(root, context) for root in errors)
        ):
            error.__context__ = None
        elif context is not None:
            pending.append(context)
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if _is_exception_group(error):
            pending.extend(cast("_ExceptionGroupProtocol", error).exceptions)


def _same_exception_tree(first: BaseException, second: BaseException) -> bool:
    pending = [(first, second)]
    seen: set[tuple[int, int]] = set()
    while pending:
        first_error, second_error = pending.pop()
        if first_error is second_error:
            continue
        if (pair := (id(first_error), id(second_error))) in seen:
            continue
        seen.add(pair)
        if (
            type(first_error) is not type(second_error)
            or not _is_exception_group(first_error)
            or not _is_exception_group(second_error)
        ):
            return False
        first_group = cast("_ExceptionGroupProtocol", first_error)
        second_group = cast("_ExceptionGroupProtocol", second_error)
        if first_group.message != second_group.message or len(first_group.exceptions) != len(second_group.exceptions):
            return False
        pending.extend(zip(first_group.exceptions, second_group.exceptions, strict=True))
    return True


def _contains_exception(error: BaseException, target: BaseException | None) -> bool:
    if target is None or not _is_exception_group(error):
        return False
    pending = list(cast("_ExceptionGroupProtocol", error).exceptions)
    seen: set[int] = set()
    while pending:
        child = pending.pop()
        if child is target:
            return True
        if id(child) in seen:
            continue
        seen.add(id(child))
        if _is_exception_group(child):
            pending.extend(cast("_ExceptionGroupProtocol", child).exceptions)
    return False


def _append_exception_context(error: BaseException, context: BaseException) -> None:
    if _exception_graph_contains(error, context) or _exception_graph_contains(context, error):
        return
    if error.__context__ is None:
        error.__context__ = context
        return
    tail = error
    seen: set[int] = set()
    while id(tail) not in seen:
        seen.add(id(tail))
        if (next_error := tail.__cause__ if tail.__cause__ is not None else tail.__context__) is None:
            tail.__context__ = context
            return
        tail = next_error


def _exception_graph_contains(error: BaseException, target: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        if id(current) in seen:  # pragma: no cover - arbitrary caller exceptions can contain cycles
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if _is_exception_group(current):
            pending.extend(cast("_ExceptionGroupProtocol", current).exceptions)
    return False


def _grouped_errors(
    error: BaseException, message: str, marker: tuple[str, _MarkerValue]
) -> tuple[BaseException, ...] | None:
    if not _is_exception_group(error):
        return None
    group = cast("_ExceptionGroupProtocol", error)
    return group.exceptions if group.message == message and getattr(group, marker[0], None) is marker[1] else None


if TYPE_CHECKING:

    class _ExceptionGroupProtocol(Protocol):
        @property
        def message(self) -> str: ...

        @property
        def exceptions(self) -> tuple[BaseException, ...]: ...


def _raise_chained_errors(first_error: BaseException, second_error: BaseException | None = None) -> NoReturn:
    if second_error is None:
        first_context = first_error.__context__
        try:
            raise first_error  # ruff:ignore[raise-within-try]  # the handler restores caller-supplied context before propagation
        except BaseException:
            first_error.__context__ = first_context
            raise
    if (second_context := second_error.__context__) is not None and second_context is not first_error:
        _detach_exception_context(second_context, first_error)
        _append_exception_context(first_error, second_context)
    first_context = first_error.__context__
    try:
        raise first_error  # ruff:ignore[raise-within-try]  # the second raise needs this error as implicit context
    except BaseException:  # ruff:ignore[blind-except]  # first_error may be a control-flow exception
        first_error.__context__ = first_context
        try:
            raise second_error  # ruff:ignore[raise-within-try]  # the handler makes the chain interpreter-independent
        except BaseException:
            second_error.__context__ = first_error
            first_error.__context__ = first_context
            raise


def _detach_exception_context(error: BaseException, target: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__context__ is target:
            current.__context__ = None
        elif current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if _is_exception_group(current):
            pending.extend(cast("_ExceptionGroupProtocol", current).exceptions)


def _raise_body_and_release(body_error: BaseException, release_error: BaseException) -> NoReturn:
    # Group mode: surface the body failure and the release failure as sibling leaves instead of letting one hide in the
    # other's __context__. BaseExceptionGroup returns a plain ExceptionGroup when both leaves subclass Exception, so
    # ``except*`` and ``except Exception`` still catch them; a BaseException leaf (KeyboardInterrupt, CancelledError)
    # keeps the group outside ordinary handlers. ``from None`` stops the group itself gaining a redundant __context__.
    _raise_grouped_errors("lock body and release both failed", body_error, release_error)


def _raise_cleanup_errors(
    message: str,
    primary_error: BaseException,
    *cleanup_errors: BaseException | None,
) -> NoReturn:
    _raise_grouped_errors(
        message,
        primary_error,
        *(error for error in cleanup_errors if error is not None),
    )


# On Windows os.path.realpath calls CreateFileW with share_mode=0, which blocks concurrent DeleteFileW and causes
# livelocks under threaded contention with SoftFileLock. os.path.abspath is purely string-based and avoids this.
_resolve_dir: Final[Callable[[str], str]] = os.path.abspath if sys.platform == "win32" else os.path.realpath


def _canonical(path: str | os.PathLike[str]) -> str:
    """
    Return one stable key for *path*, collapsing equivalent spellings without following a final symlink.

    Relative, absolute, and ``./`` spellings of one lock file must map to a single singleton instance, deadlock-registry
    entry, and removal key. Resolving the whole path with ``realpath`` would follow a final symlink and alias a lock
    target the backend deliberately rejects, so the registry identity would differ from the backend's. Resolving only
    the parent directory and re-appending the literal final component collapses the equivalent spellings while keeping a
    final symlink a distinct key. On Windows the parent is resolved with ``abspath`` so junctions and reparse points are
    not followed either.
    """
    parent, name = os.path.split(os.fspath(path))
    return os.path.join(_resolve_dir(parent or os.curdir), name)  # ruff:ignore[os-path-join]  # string join matches abspath/realpath


class _ThreadLocalRegistry(local):
    def __init__(self) -> None:
        super().__init__()
        self.held: dict[Hashable, int] = {}


_registry: Final[_ThreadLocalRegistry] = _ThreadLocalRegistry()


_T = TypeVar("_T", bound="BaseFileLock")
_LockT = TypeVar("_LockT", bound="BaseFileLock | ReadWriteLock | SoftReadWriteLock")


class FileLockMeta(ABCMeta):
    _instances: WeakValueDictionary[str, BaseFileLock]
    _instances_lock: RLock
    _instances_under_construction: set[str]

    def __call__(  # ruff:ignore[too-many-arguments]  # forwards the public constructor's documented parameters
        cls: type[_T],
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = _UNSET_FILE_MODE,
        thread_local: bool = True,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        poll_interval: float = 0.05,
        lifetime: float | None = None,
        context_error_policy: ContextErrorPolicy = "chain",
        close_error_policy: CloseErrorPolicy = "default",
        fallback_to_soft: bool = True,
        preserve_lock_file: bool = False,
        on_acquired: Callable[[int], None] | None = None,
        **kwargs: _ExtraValue,
    ) -> _T:
        _ensure_current_process()
        lifetime = _resolve_lifetime(lifetime, cls, stacklevel=cls._constructor_lifetime_warning_stacklevel)
        poll_interval = _resolve_poll_interval(poll_interval)
        # Validate before building the instance: a raise inside __init__ would leave a half-constructed object whose
        # __del__ then trips over the missing context.
        _check_blocking(blocking)
        # nan is left to acquire, which knows whether the call blocks; a non-blocking call ignores the timeout.
        timeout = _seconds("timeout", timeout)
        _check_mode(mode, cls.__name__)
        context_error_policy = _resolve_context_error_policy(context_error_policy)
        close_error_policy = _resolve_close_error_policy(close_error_policy)
        preserve_lock_file = _resolve_preserve_lock_file(
            preserve=preserve_lock_file, supported=cls._preserve_lock_file_supported, cls_name=cls.__name__
        )
        on_acquired = _resolve_on_acquired(on_acquired, supported=cls._on_acquired_supported, cls_name=cls.__name__)
        # Checked before the singleton lookup so a cache hit rejects a misspelled option just like a fresh build does.
        params: Final = _supported_init_params(
            cls,
            {
                "timeout": timeout,
                "mode": mode,
                "thread_local": thread_local,
                "blocking": blocking,
                "is_singleton": is_singleton,
                "poll_interval": poll_interval,
                "lifetime": lifetime,
                "context_error_policy": context_error_policy,
                "close_error_policy": close_error_policy,
                "fallback_to_soft": fallback_to_soft,
                "preserve_lock_file": preserve_lock_file,
                "on_acquired": on_acquired,
                **kwargs,
            },
        )
        if not is_singleton:
            return super().__call__(lock_file, **params)

        # Look up, build and store under one lock. Without it two threads racing the first construction for a
        # path both miss the cache and each build their own instance, so callers relying on is_singleton for
        # reentrant locking across instances end up with two "singletons" and acquire()'s deadlock check then
        # rejects a legitimate reentrant acquire; the unguarded writes to the WeakValueDictionary are a data
        # race besides. ReadWriteLock and SoftReadWriteLock already guard their singleton caches this way.
        # Key the cache on the canonical form so equivalent spellings of one path share a singleton, and it matches the
        # deadlock-registry key acquire() uses.
        singleton_key = _canonical(lock_file)
        with cls._instances_lock:
            if (instance := cls._instances.get(singleton_key)) is None:
                if singleton_key in cls._instances_under_construction:  # pragma: needs fork
                    msg = f"Singleton lock construction is already active for {lock_file!s}"
                    raise RuntimeError(msg)
                construction_registry = cls._instances_under_construction
                construction_pid = os.getpid()
                construction_registry.add(singleton_key)
                try:
                    instance = super().__call__(lock_file, **params)
                finally:
                    construction_registry.discard(singleton_key)
                if os.getpid() != construction_pid:  # pragma: needs fork
                    msg = "Lock construction cannot continue after fork; construct a new lock in the child"
                    raise RuntimeError(msg)
                cls._instances[singleton_key] = instance
                return instance

        non_matching_params = {
            name: (str(passed_param), str(set_param))
            for name, (passed_param, set_param) in {
                "thread_local": (thread_local, instance.is_thread_local()),
                # Compare the wait, not the value: nan never equals itself, and -1, -5 and inf all mean no limit.
                "timeout": (
                    _timeout_behavior(timeout, any_negative=True),
                    _timeout_behavior(instance.timeout, any_negative=True),
                ),
                "mode": (mode, instance._context.mode),  # ruff:ignore[private-member-access]  # compares against the managed instance's own context
                "blocking": (blocking, instance.blocking),
                "poll_interval": (poll_interval, instance.poll_interval),
                "lifetime": (lifetime, instance.lifetime),
                "context_error_policy": (context_error_policy, instance.context_error_policy),
                "close_error_policy": (close_error_policy, instance.close_error_policy),
                "fallback_to_soft": (fallback_to_soft, instance.fallback_to_soft),
                "preserve_lock_file": (preserve_lock_file, instance.preserve_lock_file),
            }.items()
            if passed_param != set_param
        }
        # The metaclass owns the instance it returns, so reading the subclass hook stays inside this module's contract.
        non_matching_params.update(instance._singleton_extra_mismatches(kwargs))  # ruff:ignore[private-member-access]
        # Callables compare by identity, not equality: two equal callables can close over different state, so a
        # singleton must reject a different hook object even if it compares equal. Keep it out of the scalar dict above.
        hook_mismatch = on_acquired is not instance.on_acquired
        if not non_matching_params and not hook_mismatch:
            return instance  # ty: ignore[invalid-return-type]  # https://github.com/astral-sh/ty/issues/3231

        msg = "Singleton lock instances cannot be initialized with differing arguments"
        msg += "\nNon-matching arguments: "
        for param_name, (passed_param, set_param) in non_matching_params.items():
            msg += f"\n\t{param_name} (existing lock has {set_param} but {passed_param} was passed)"
        if hook_mismatch:
            msg += f"\n\ton_acquired (existing lock has {instance.on_acquired} but {on_acquired} was passed)"
        raise ValueError(msg)


def _resolve_lifetime(lifetime: float | None, cls: type[BaseFileLock], *, stacklevel: int) -> float | None:
    """
    Validate ``lifetime`` and drop a value the backend cannot honor.

    ``lifetime`` is a deliberate age-based lease: a lock file older than ``lifetime`` is broken even while its holder is
    still alive. Existence locks (:class:`SoftFileLock`) implement that behavior by unlinking a reclaimable pathname,
    which can overlap a live holder. A native OS lock lives on the inode, so unlinking the pathname by age cannot revoke
    the kernel lock; a contender would lock a fresh inode and overlap the live holder (#590). Ignore the request with a
    warning rather than accept a setting that breaks mutual exclusion.
    """
    if lifetime is None:
        return None
    # An int stays exact: no float holds every int 4.0.12 accepted here.
    if isinstance(lifetime, bool) or not isinstance(lifetime, int):
        lifetime = _seconds("lifetime", lifetime)
    if lifetime < 0 or (isinstance(lifetime, float) and not math.isfinite(lifetime)):
        msg = f"lifetime must be finite and non-negative, not {lifetime!r}"
        raise ValueError(msg)
    if not cls._lifetime_supported:
        warnings.warn(
            f"lifetime is ignored for {cls.__name__}: {cls._lifetime_unsupported_reason}; "
            f"only SoftFileLock supports lifetime-based expiry",
            stacklevel=stacklevel,
        )
        return None
    if cls._lifetime_replacements is not None:
        strict_lock, lease = cls._lifetime_replacements
        warnings.warn(
            f"{cls.__name__}(lifetime=...) uses age-based expiry and can overlap a live holder; "
            f"use {lease} for expiry or {strict_lock} for fail-closed locking",
            SoftFileLockLifetimeWarning,
            stacklevel=stacklevel,
        )
    return lifetime


def _resolve_poll_interval(poll_interval: float) -> float:
    """Validate ``poll_interval`` used for ``time.sleep`` / ``asyncio.sleep`` retries."""
    if (poll_interval := _seconds("poll_interval", poll_interval)) < 0 or not math.isfinite(poll_interval):
        msg = f"poll_interval must be finite and non-negative, not {poll_interval!r}"
        raise ValueError(msg)
    _check_timeout_max("poll_interval", poll_interval)
    return poll_interval


def _check_blocking(blocking: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]  # validates the option's own value
    if not isinstance(blocking, bool):
        msg = f"blocking must be a bool, not {type(blocking).__name__}"
        raise TypeError(msg)


def _seconds(name: str, value: float) -> float:
    # 4.0.12 accepted Decimal and Fraction; a bool is a typo, and a string would only fail later.
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        msg = f"{name} must be a number of seconds, not {type(value).__name__}"
        raise TypeError(msg)
    # float() refuses a signaling NaN, which means no number of seconds just as a quiet one does.
    if isinstance(value, Decimal) and value.is_snan():
        return math.nan
    try:
        seconds: Final = float(value)
    except OverflowError:
        msg = f"{name} is too large to be a number of seconds: {value!r}"
        raise ValueError(msg) from None
    # float() rounds a tiny negative to -0.0, which compares as zero, yet timeout and lifetime checks read the sign.
    return -math.ulp(0.0) if seconds == 0 and value < 0 else seconds


def _check_mode(mode: int, cls_name: str) -> None:
    if mode == _UNSET_FILE_MODE:
        return
    if isinstance(mode, bool) or not isinstance(mode, int):
        msg = f"mode must be an int, not {type(mode).__name__}"
        raise TypeError(msg)
    # A lock file is never executed, so setuid, setgid and sticky have no meaning on it and only widen its permissions.
    if not 0 <= mode <= _PERMISSION_BITS:
        msg = f"mode={mode:#o} must be within 0o000-0o777 for {cls_name}"
        raise ValueError(msg)
    # A lock reopens, reads, or deletes the files it creates, so a mode that denies the owner read or write fails later
    # and for good.
    if ~mode & (stat.S_IRUSR | stat.S_IWUSR):
        msg = f"mode={mode:#o} must grant the owner read and write for {cls_name}"
        raise ValueError(msg)


def _resolve_context_error_policy(policy: str) -> ContextErrorPolicy:
    if policy not in _CONTEXT_ERROR_POLICIES:
        msg = f"context_error_policy must be 'chain' or 'group', got {policy!r}"
        raise ValueError(msg)
    # Fail fast at construction rather than only when a dual failure happens to occur.
    if policy == "group" and _exception_group_cls() is None:  # pragma: <3.11 cover
        msg = "context_error_policy='group' requires Python 3.11+ or the 'exceptiongroup' backport installed"
        raise ValueError(msg)
    return cast("ContextErrorPolicy", policy)


def _resolve_close_error_policy(policy: str) -> CloseErrorPolicy:
    if policy not in _CLOSE_ERROR_POLICIES:
        msg = f"close_error_policy must be 'default', 'raise', or 'suppress', got {policy!r}"
        raise ValueError(msg)
    return cast("CloseErrorPolicy", policy)


def _resolve_preserve_lock_file(*, preserve: bool, supported: bool, cls_name: str) -> bool:
    # An existence lock unlinks its marker to release, so preserving the pathname would defeat unlocking. Reject the
    # request rather than silently ignore it, since a caller asking for a stable identity must know it cannot be kept.
    if preserve and not supported:
        msg = f"preserve_lock_file=True is not supported by {cls_name}: unlinking its marker is how it releases"
        raise ValueError(msg)
    return preserve


def _resolve_on_acquired(
    on_acquired: Callable[[int], None] | None, *, supported: bool, cls_name: str
) -> Callable[[int], None] | None:
    if on_acquired is None:
        return None
    # An existence lock stores protocol state in its marker, so a caller writing through the descriptor would corrupt
    # stale detection and ownership metadata; only native locks lend out the descriptor.
    if not supported:
        msg = f"on_acquired is not supported by {cls_name}: only native locks expose the lock descriptor"
        raise ValueError(msg)
    return on_acquired


def _supported_init_params(
    cls: type[BaseFileLock], params: dict[str, _LockInitValue | _ExtraValue]
) -> dict[str, _LockInitValue | _ExtraValue]:
    if (model := _init_parameter_model(cls)).accepts_kwargs:
        return params
    if unsupported := sorted(
        name
        for name, value in params.items()
        if name not in model.accepted_params
        and ((parameter := model.default_params.get(name)) is None or value != parameter.default)
    ):
        msg = f"{cls.__name__} does not support non-default lock options: {', '.join(unsupported)}"
        raise TypeError(msg)
    # virtualenv narrows a BaseFileLock descendant's signature; omit base defaults it does not accept (#340).
    return {name: value for name, value in params.items() if name in model.accepted_params}


def _timeout_behavior(timeout: float, *, any_negative: bool) -> float | str:
    # inf, and for a BaseFileLock any negative value, wait without a limit like -1.
    if math.isnan(timeout):
        return "nan"
    return -1.0 if timeout == math.inf or (any_negative and timeout < 0) else timeout


def _check_timeout_max(name: str, value: float) -> None:
    # Lock.acquire, Event.wait and time.sleep raise OverflowError past this bound, which is only ~49.7 days on Windows.
    if value > TIMEOUT_MAX:
        msg = f"{name} must not exceed threading.TIMEOUT_MAX ({TIMEOUT_MAX}), got {value!r}"
        raise ValueError(msg)


_INIT_PARAMETER_MODELS: Final[WeakKeyDictionary[type[BaseFileLock], _InitParameterModel]] = WeakKeyDictionary()


def _init_parameter_model(cls: type[BaseFileLock]) -> _InitParameterModel:
    # A strong cache would keep dynamically created subclasses alive for the process lifetime.
    with _fork_transition(), _FORK_STATE.parameter_models_lock:
        if (model := _INIT_PARAMETER_MODELS.get(cls)) is None:
            parameters = inspect.signature(cls.__init__).parameters.values()
            model = _InitParameterModel(
                accepted_params=frozenset(
                    parameter.name
                    for parameter in parameters
                    if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
                ),
                accepts_kwargs=any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters),
                default_params={
                    name: parameter
                    for name, parameter in inspect.signature(type(cls).__call__).parameters.items()
                    if parameter.default is not inspect.Parameter.empty
                },
            )
            _INIT_PARAMETER_MODELS[cls] = model
        return model


@dataclass(frozen=True)
class _InitParameterModel:
    accepted_params: frozenset[str]
    accepts_kwargs: bool
    default_params: dict[str, inspect.Parameter]


class BaseFileLock(contextlib.ContextDecorator, metaclass=FileLockMeta):  # ruff:ignore[too-many-public-methods]  # public config properties
    """
    Abstract base class for a file lock object.

    Provides the common reentrant API and state management. Subclasses implement the locking mechanism
    (:class:`UnixFileLock <filelock.UnixFileLock>`, :class:`WindowsFileLock <filelock.WindowsFileLock>`,
    :class:`SoftFileLock <filelock.SoftFileLock>`).

    """

    _instances: WeakValueDictionary[str, BaseFileLock]
    _instances_lock: RLock
    _instances_under_construction: set[str]

    #: How the cross-instance deadlock message names the conflicting holder; the async subclass says "task".
    _deadlock_holder_desc: str = "FileLock instance in this thread"

    #: Whether an age-based :attr:`lifetime` lease may break this lock. Only existence locks set it (they reclaim by
    #: unlinking a pathname); native OS locks leave it ``False`` since a kernel lock cannot be revoked by file age.
    _lifetime_supported: bool = False

    #: Strict-lock and lease replacements for a backend with legacy age-based expiry.
    _lifetime_replacements: tuple[str, str] | None = None

    #: Why a backend that refuses ``lifetime`` cannot honor it, named in the warning that drops the value.
    _lifetime_unsupported_reason: str = "a native OS lock cannot be broken safely by file age"

    #: Async construction adds one metaclass frame before lifetime validation.
    _constructor_lifetime_warning_stacklevel: int = 3

    #: Whether :attr:`preserve_lock_file` may be ``True``. Native locks keep the pathname on release, so they support
    #: it; existence locks unlink their marker to release and reject it.
    _preserve_lock_file_supported: bool = True

    #: Whether an :attr:`on_acquired` hook may be set. Native locks lend the descriptor out; existence locks keep
    #: protocol state in the marker and reject it.
    _on_acquired_supported: bool = True

    #: Ceiling in seconds on the jittered backoff between contended acquisition retries. ``0`` keeps the fixed
    #: poll cadence; a multi-file backend sets it so contending processes desynchronize instead of livelocking.
    _poll_backoff_cap: float = 0.0

    def __init_subclass__(cls, **kwargs: _SubclassValue) -> None:
        """Give each lock subclass its own singleton registry and lock."""
        super().__init_subclass__(**kwargs)
        cls._instances = WeakValueDictionary()
        cls._instances_lock = RLock()
        cls._instances_under_construction = set()
        _register_fork_class(cls)

    @classmethod
    def _reset_class_after_fork(cls) -> None:  # pragma: forked child
        cls._instances = WeakValueDictionary()
        cls._instances_lock = RLock()
        cls._instances_under_construction = set()

    def __init__(  # ruff:ignore[too-many-arguments]  # public constructor: one parameter per documented lock option
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = _UNSET_FILE_MODE,
        thread_local: bool = True,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        poll_interval: float = 0.05,
        lifetime: float | None = None,
        context_error_policy: ContextErrorPolicy = "chain",
        close_error_policy: CloseErrorPolicy = "default",
        fallback_to_soft: bool = True,
        preserve_lock_file: bool = False,
        on_acquired: Callable[[int], None] | None = None,
    ) -> None:
        """
        Create a new lock object.

        :param lock_file: path to the file
        :param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in the
            acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it to a
            negative value or ``inf``. A timeout of 0 means that there is exactly one attempt to acquire the file lock.
        :param mode: file permissions for the lockfile. When not specified, the OS controls permissions via umask and
            default ACLs, preserving POSIX default ACL inheritance in shared directories. Construction raises
            :class:`ValueError` unless it grants the owner read and write, which the lock needs for its own files.
        :param thread_local: Whether this object's internal context should be thread local or not. If this is set to
            ``False`` then the lock will be reentrant across threads. When ``True`` (the default), **all fields of the
            lock's internal context are per-thread**, including the configuration values ``poll_interval``, ``timeout``,
            ``blocking``, ``mode``, and ``lifetime``. Setting one of these properties from one thread does not change
            the value seen by another thread; threads that did not perform the write continue to see the value supplied
            at construction time. ``mode`` has no setter, so construction is the only place it is ever set. If you need
            configuration values to be visible across threads, construct the lock with ``thread_local=False``.
        :param blocking: whether the lock should be blocking or not
        :param is_singleton: If this is set to ``True`` then only one instance of this class will be created per lock
            file. This is useful if you want to use the lock object for reentrant locking without needing to pass the
            same object around.
        :param poll_interval: default interval for polling the lock file, in seconds. It will be used as fallback value
            in the acquire method, if no poll_interval value (``None``) is given.
        :param lifetime: for :class:`SoftFileLock`, the age in seconds after which a waiting process may delete the
            marker, even while its holder remains alive. This legacy expiry mode does not provide strict mutual
            exclusion. ``None`` (the default) disables age-based expiry. Native OS locks (:class:`FileLock`) cannot be
            revoked by file age and ignore a non-``None`` ``lifetime`` with a warning.
        :param context_error_policy: how a context manager reconciles a failure in its body with a failure while
            releasing on exit. ``"chain"`` (the default) keeps Python's behavior: the release error propagates with the
            body error in its ``__context__``. ``"group"`` raises a :class:`BaseExceptionGroup` holding the body error
            first and the release error second, so neither hides the other.
        :param close_error_policy: what to do with an ``os.close`` failure after relinquishing descriptor ownership.
            ``"default"`` keeps each backend's historical behavior (Unix native locks drop a FUSE/Docker ``EIO``;
            Windows native locks and :class:`SoftFileLock` propagate); ``"raise"`` always propagates the ``OSError``;
            ``"suppress"`` always ignores it. Held state is released either way. It does not affect unlock failures or
            lock-file deletion.
        :param fallback_to_soft: for :class:`UnixFileLock`, whether to switch to :class:`SoftFileLock` when the
            filesystem's ``flock`` returns ``ENOSYS``. ``True`` (the default) keeps the historical fallback;
            ``False`` fails closed, letting the ``ENOSYS`` propagate so a caller that needs kernel-enforced
            locking is never silently downgraded. It has no effect on Windows or :class:`SoftFileLock`.
        :param preserve_lock_file: for native locks (:class:`FileLock`), whether filelock promises not to unlink the
            lock pathname on release. ``False`` (the default) keeps each backend's cleanup: Windows removes the lock
            file, Unix already leaves it. ``True`` keeps a stable file identity for ACLs, auditing, and holder metadata:
            Windows skips its post-release unlink and Unix refuses to enter the ``ENOSYS`` soft fallback (which releases
            by unlinking). :class:`SoftFileLock` rejects ``True``. The promise covers filelock's own release path only;
            it cannot stop another process or the filesystem from removing the pathname.
        :param on_acquired: for native locks (:class:`FileLock`), a callable invoked with the borrowed lock descriptor
            once per physical acquisition, after filelock holds the native lock and finished backend initialization but
            before :meth:`~BaseFileLock.acquire` returns. Recursive acquisitions do not call it again. The callback may
            read, write, seek, truncate, or set metadata through ``os`` on the descriptor, but must not close, unlock,
            or take ownership of it, and filelock does not fsync its writes. If it raises, filelock releases the lock
            and re-raises. :class:`SoftFileLock` rejects the hook.

        :raises TypeError: if *timeout* or *poll_interval* is a :class:`bool` or not a real number, *blocking* is not a
            :class:`bool`, or *mode* is not an :class:`int`
        :raises ValueError: if *poll_interval* is negative, not finite, or above :data:`threading.TIMEOUT_MAX`, if
            *timeout* is too large to convert to :class:`float`, or if *mode* falls outside ``0o000``-``0o777`` or
            denies the owner read and write

        """
        self._creator_pid = os.getpid()
        self._transition_lock = RLock()
        self._is_thread_local = thread_local
        self._is_singleton = is_singleton
        self._context_error_policy = context_error_policy  # already validated by the metaclass
        self._close_error_policy = close_error_policy  # already validated by the metaclass
        self._fallback_to_soft = fallback_to_soft
        self._preserve_lock_file = preserve_lock_file  # already validated by the metaclass
        self._on_acquired = on_acquired  # already validated by the metaclass

        self._context: FileLockContext = (ThreadLocalFileContext if thread_local else FileLockContext)(
            lock_file=os.fspath(lock_file),
            timeout=timeout,
            mode=mode,
            blocking=blocking,
            poll_interval=poll_interval,
            lifetime=lifetime,
        )
        #: Each thread's committed hold on a thread-local lock, so another thread can release it.
        self._thread_holds: dict[int, tuple[FileLockContext, Thread]] = {}
        #: Each thread's uncommitted acquisition, so a backend can finish it for an exited thread.
        self._thread_pending: dict[int, tuple[FileLockContext, Thread]] = {}
        _register_fork_object(self)

    def _singleton_extra_mismatches(  # ruff:ignore[no-self-use]  # the base class adds no options of its own
        self, _kwargs: Mapping[str, _ExtraValue], /
    ) -> dict[str, tuple[str, str]]:
        # A subclass's own options reach the metaclass only as kwargs, and only the subclass knows their defaults and
        # how __init__ resolves them, so each override compares its options and merges what super() reports.
        return {}

    def is_thread_local(self) -> bool:
        """:returns: a flag indicating if this lock is thread local or not"""
        return self._is_thread_local

    @property
    def is_singleton(self) -> bool:
        """
        A flag indicating if this lock is singleton or not.

        .. versionadded:: 3.13.0

        """
        return self._is_singleton

    @property
    def context_error_policy(self) -> ContextErrorPolicy:
        """
        How a context manager reconciles a body failure with a release failure on exit.

        .. versionadded:: 3.30.0

        """
        return self._context_error_policy

    @property
    def close_error_policy(self) -> CloseErrorPolicy:
        """
        What a lock does with an ``os.close`` failure after relinquishing descriptor ownership.

        .. versionadded:: 3.30.0

        """
        return self._close_error_policy

    def _close_released_fd(self, fd: int, *, default_suppresses: bool) -> None:
        # CPython never retries close() after EINTR because the descriptor number may already be reused, so neither does
        # this. close_error_policy decides the error's fate after the backend relinquishes descriptor ownership.
        try:
            os.close(fd)
        except OSError:
            if not self._suppresses_close_error(default_suppresses=default_suppresses):
                raise

    def _suppresses_close_error(self, *, default_suppresses: bool) -> bool:
        return self._close_error_policy == "suppress" or (self._close_error_policy == "default" and default_suppresses)

    @property
    def fallback_to_soft(self) -> bool:
        """
        Whether a :class:`FileLock` falls back to :class:`SoftFileLock` when the filesystem lacks ``flock``.

        Only :class:`UnixFileLock` acts on it: when ``False`` an ``ENOSYS`` from ``flock`` propagates instead of
        switching to existence-lock semantics.

        .. versionadded:: 3.30.0

        """
        return self._fallback_to_soft

    @property
    def preserve_lock_file(self) -> bool:
        """
        Whether filelock promises not to unlink the lock pathname on release.

        When ``True``, Windows skips its post-release unlink and Unix refuses the ``ENOSYS`` soft fallback.
        :class:`SoftFileLock` rejects ``True`` because unlinking its marker is how it releases.

        .. versionadded:: 3.30.0

        """
        return self._preserve_lock_file

    @property
    def on_acquired(self) -> Callable[[int], None] | None:
        """
        The callback run with the borrowed lock descriptor once per physical acquisition, or ``None``.

        Native locks only. It runs after the native lock is held and backend initialization finished, before
        :meth:`~BaseFileLock.acquire` returns; a raise rolls back the acquisition. :class:`SoftFileLock` rejects it.

        .. versionadded:: 3.30.0

        """
        return self._on_acquired

    @property
    def lock_file(self) -> str:
        """Path to the lock file."""
        return self._context.lock_file

    @property
    def timeout(self) -> float:
        """
        The default timeout value, in seconds.

        .. versionadded:: 2.0.0

        """
        return self._context.timeout

    @timeout.setter
    def timeout(self, value: float | str | bytes | bytearray) -> None:
        """
        Change the default timeout value.

        :param value: the new value, in seconds; a negative value or ``inf`` waits without a limit

        :raises TypeError: if *value* is a :class:`bool` or not a real number
        :raises ValueError: if *value* is too large to convert to :class:`float`

        .. deprecated:: 4.0.13

            Passing a numeric string; pass the number instead.

        """
        if isinstance(value, (str, bytes, bytearray)):
            warnings.warn(
                "setting timeout from a string is deprecated and will be removed; pass a number",
                DeprecationWarning,
                stacklevel=2,
            )
            value = float(value)
        self._context.timeout = _seconds("timeout", value)

    @property
    def blocking(self) -> bool:
        """
        Whether the locking is blocking or not.

        .. versionadded:: 3.14.0

        """
        return self._context.blocking

    @blocking.setter
    def blocking(self, value: bool) -> None:
        """
        Change the default blocking value.

        :param value: the new value as bool

        :raises TypeError: if *value* is not a bool

        """
        _check_blocking(value)
        self._context.blocking = value

    @property
    def poll_interval(self) -> float:
        """
        The default polling interval, in seconds.

        .. versionadded:: 3.24.0

        """
        return self._context.poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        """
        Change the default polling interval.

        :param value: the new value, in seconds

        :raises ValueError: if *value* is negative, not finite, or above :data:`threading.TIMEOUT_MAX`
        :raises TypeError: if *value* is a :class:`bool` or not a real number

        """
        self._context.poll_interval = _resolve_poll_interval(value)

    @property
    def lifetime(self) -> float | None:
        """
        The soft marker age in seconds that permits expiry, or ``None`` to disable age-based expiry.

        A non-``None`` value permits a waiter to enter while the previous holder remains active, so it does not provide
        strict mutual exclusion. Native locks ignore the value with a warning.

        .. versionadded:: 3.24.0

        """
        return self._context.lifetime

    @lifetime.setter
    def lifetime(self, value: float | None) -> None:
        """
        Change the legacy age-based expiry threshold.

        :param value: the new value in seconds, or ``None`` to disable expiration

        :raises ValueError: if *value* is negative or not finite
        :raises TypeError: if *value* is not ``None`` and not a real number

        """
        self._context.lifetime = _resolve_lifetime(value, type(self), stacklevel=3)

    @property
    def mode(self) -> int:
        """
        The file permissions requested for the lock file.

        Without an explicit ``mode`` this reports ``0o644`` for backwards compatibility, but the file is created with
        ``0o666`` filtered through the umask and default ACLs, so with a ``002`` umask it ends up ``0o664``. Check
        :attr:`has_explicit_mode` before relying on the value, or ``os.stat`` the file for its actual permissions.

        """
        return 0o644 if self._context.mode == _UNSET_FILE_MODE else self._context.mode

    @property
    def has_explicit_mode(self) -> bool:
        """Whether the file permissions were explicitly set."""
        return self._context.mode != _UNSET_FILE_MODE

    def _open_mode(self) -> int:
        """Mode for ``os.open``: 0o666 when unset so umask and ACLs decide, otherwise the explicit mode."""
        return 0o666 if self._context.mode == _UNSET_FILE_MODE else self._context.mode

    @property
    def is_locked(self) -> bool:
        """
        A boolean indicating if the lock file is holding the lock currently.

        .. versionchanged:: 2.0.0

            This was previously a method and is now a property.

        """
        _ensure_current_process()
        return self._context.lock_file_fd is not None

    @property
    def lock_counter(self) -> int:
        """The number of times this lock has been acquired (but not yet released)."""
        _ensure_current_process()
        return self._context.lock_counter

    def __enter__(self) -> Self:
        """
        Acquire the lock.

        :returns: the lock object

        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock, reconciling a release failure with any body failure per :attr:`context_error_policy`."""
        self._release_in_context(exc_value)

    def _release_in_context(self, body_error: BaseException | None) -> None:
        # Release from a context-manager exit. "chain" lets a release failure propagate with the body error already in
        # its __context__ (Python's default); "group" raises both as sibling leaves so neither one hides the other.
        try:
            self.release()
        except BaseException as release_error:
            if body_error is None or self._context_error_policy == "chain":
                raise
            _raise_body_and_release(body_error, release_error)

    def __del__(self) -> None:
        """Force-release so a dropped reference never leaks a held lock, including every thread's hold."""
        if vars(self).get("_creator_pid") != os.getpid():
            return  # pragma: forked child
        # A finalizer must not raise. A release error during garbage collection would otherwise surface as an
        # unraisable-exception warning, attributed to whichever code triggered collection. The dropped lock still gets
        # best-effort cleanup; an explicit release() reports the same error to a caller who can act on it.
        with contextlib.suppress(Exception):
            self.release(force=True)
        # Nothing references the lock any more, so no thread can still be inside it through this instance.
        with contextlib.suppress(Exception):
            self._release_thread_holds(exited_only=False)

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        poll_intervall: float | None = None,
        blocking: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> AcquireReturnProxy[Self]:
        """
        Try to acquire the file lock.

        :param timeout: maximum wait time for acquiring the lock, ``None`` means use the default :attr:`~timeout` is and
            if ``timeout < 0``, there is no timeout and this method will block until the lock could be acquired
        :param poll_interval: interval of trying to acquire the lock file, ``None`` means use the default
            :attr:`~poll_interval`
        :param poll_intervall: deprecated, kept for backwards compatibility, use ``poll_interval`` instead
        :param blocking: whether to retry until this call holds the lock or the timeout expires, ``None`` means use
            the default :attr:`~blocking`. ``False`` makes one attempt and raises :class:`~Timeout` on failure.
        :param cancel_check: a callable returning ``True`` when the acquisition should be canceled. Checked on each poll
            iteration. When triggered, raises :class:`~Timeout` just like an expired timeout.

        :returns: a context object that will unlock the file when the context is exited

        :raises Timeout: if fails to acquire lock within the timeout period
        :raises ValueError: if a blocking call gets a ``nan`` timeout, or *poll_interval* exceeds
            :data:`threading.TIMEOUT_MAX`

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            with lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            lock.acquire()
            try:
                pass
            finally:
                lock.release()

        .. versionchanged:: 2.0.0

            This method returns now a *proxy* object instead of *self*, so that it can be used in a with statement
            without side effects.

        """
        self._raise_if_inherited()
        if timeout is None:
            timeout = self._context.timeout

        if blocking is None:
            blocking = self._context.blocking

        if poll_intervall is not None:
            msg = "use poll_interval instead of poll_intervall"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            poll_interval = poll_intervall

        poll_interval = poll_interval if poll_interval is not None else self._context.poll_interval
        poll_interval = _resolve_poll_interval(poll_interval)
        timeout = _resolve_timeout(timeout, blocking=blocking)

        start_time = time.perf_counter()
        # Wait for admission before touching any state: a caller refused entry must leave the counter, the registry and
        # the descriptor exactly as it found them.
        with self._transition_admission(
            blocking=blocking,
            cancel_check=cancel_check,
            timeout=timeout,
            poll_interval=poll_interval,
            start_time=start_time,
        ):
            # The interrupted acquire cannot run until the handler returns, so a second one here would wait on it
            # forever or take the lock out from under it. Once the lock is held, as in an on_acquired hook, it nests.
            if not self.is_locked and _on_this_stack(self._context.acquiring):
                msg = f"Deadlock: lock '{self.lock_file}' is acquired again while this thread's acquire of it waits"
                raise RuntimeError(msg)
            # Bump the counter up front; _undo_acquire rolls it back if acquisition fails.
            self._context.lock_counter += 1

            canonical = _canonical(self.lock_file)
            self._raise_if_would_deadlock(canonical, timeout=timeout, blocking=blocking)

            try:
                self._poll_until_acquired(
                    blocking=blocking,
                    cancel_check=cancel_check,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    start_time=start_time,
                )
            except BaseException:
                self._reconcile_failed_acquire(canonical)
                raise
            self._commit_acquire(canonical)
            return AcquireReturnProxy(lock=self)

    @contextlib.contextmanager
    def _transition_admission(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        timeout: float,
        poll_interval: float,
        start_time: float,
    ) -> Generator[None]:
        # One thread at a time drives the physical transition of a shared instance: its counter, descriptor and claim
        # state are plain attributes every thread reads and writes, so an unserialized loser clears the winner's
        # descriptor or double-closes it. Waiting in slices keeps cancel_check and the timeout live. A thread-local
        # context is never shared, so it skips the gate.
        if self._is_thread_local:
            yield
            return
        deadline: Final = None if timeout < 0 else start_time + timeout
        wait: Final = poll_interval if blocking else 0.0
        # wait leads the min() so a non-blocking call, whose nan timeout goes unchecked, waits 0 rather than nan.
        while not self._transition_lock.acquire(
            timeout=wait if deadline is None else min(wait, max(deadline - time.perf_counter(), 0.0))
        ):
            if not blocking or (cancel_check is not None and cancel_check()):
                raise Timeout(self.lock_file)
            if deadline is not None and time.perf_counter() >= deadline:
                raise Timeout(self.lock_file)
        try:
            yield
        finally:
            self._transition_lock.release()

    def release(self, force: bool = False) -> None:  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        """
        Release the file lock. The lock is only completely released when the lock counter reaches 0. The lock file
        itself may be deleted automatically, the behavior is platform-specific.

        :param force: If true, the lock counter is ignored and the lock is released in every case. On a thread-local
            lock it also releases the holds of threads that exited without releasing; a hold of a thread still running
            stays with that thread. Dropping the last reference to the lock releases every thread's hold.

        """
        self._release_current(force=force)
        if force:
            self._release_thread_holds(exited_only=True)

    def _release_current(self, *, force: bool) -> None:
        # A shared instance releases under the same gate its acquisition ran through, so a thread entering the lock
        # never observes a partially torn-down owner. Only an acquirer that found the lock free holds the gate for
        # long, so an unheld lock returns without waiting behind it.
        if self._context.lock_file_fd is None:
            # Only a committed hold registers, so an entry without a descriptor is a release a signal handler stopped
            # after the unlock; left alone it refuses every other instance on this path in this thread.
            if self._context.lock_file_registry is not None:
                self._finish_interrupted_release()
            return
        with contextlib.nullcontext() if self._is_thread_local else self._transition_lock:
            state: Final = self._hold_state()
            # A signal handler can land between the backend's unlock and its clearing of the descriptor, where a second
            # release would close the number twice, maybe hitting a file that reused it. The flag holds the releasing
            # frame, so only a release still on this stack stops it, and a flag a cut-short frame left set does not.
            if self._creator_pid != os.getpid() or state.lock_file_fd is None or _on_this_stack(state.releasing):
                return
            if not force and state.lock_counter > 1:
                state.lock_counter -= 1
                return
            state.releasing = inspect.currentframe()
            try:
                _LOGGER.debug("Attempting to release lock %s on %s", id(self), self.lock_file)
                self._release_with_fork_tracking()
            finally:
                # A failure after the OS unlock (during close or unlink) still released the lock: the backend cleared
                # its descriptor, so commit the counter and registry to released even as the cleanup error propagates.
                # A failure that left the lock held keeps the counter so a later release can retry the OS unlock.
                if state.lock_file_fd is None:  # pragma: no branch  # a release that returns cleared the descriptor
                    self._commit_release()
                state.releasing = None
            _LOGGER.debug("Lock %s released on %s", id(self), self.lock_file)

    def _finish_interrupted_release(self) -> None:
        # An unheld lock never waits behind an acquirer, whose own commit takes the entry over anyway.
        if not self._transition_lock.acquire(blocking=False):
            return  # pragma: no cover  # only a concurrent acquire of a shared instance holds the gate
        try:
            if self._context.lock_file_fd is None:  # pragma: no branch  # only a racing acquire fills it
                self._commit_release()
        finally:
            self._transition_lock.release()

    def _hold_state(self) -> FileLockContext:
        return self._context.current if isinstance(self._context, ThreadLocalFileContext) else self._context

    def _release_thread_holds(self, *, exited_only: bool) -> None:
        if not isinstance(context := self._context, ThreadLocalFileContext):
            return
        # Serialized, or two threads would both close an exited thread's descriptor; copy() is atomic free-threaded.
        with self._transition_lock:
            own: Final = context.current
            for held, thread in self._thread_holds.copy().values():
                if exited_only and thread.is_alive():
                    continue
                context.current = held
                try:
                    self._release_current(force=True)
                finally:
                    context.current = own

    def _raise_if_inherited(self) -> None:
        if self._creator_pid != os.getpid():  # pragma: forked child
            msg = f"{type(self).__name__} on {self.lock_file} was inherited across fork; construct a new instance"
            raise RuntimeError(msg)

    def _mark_descriptor_owned(self, fd: int, identity: tuple[int, int] | None = None) -> None:
        state: Final = self._hold_state()
        state.pending_lock_file_fd = None
        state.pending_lock_file_fd_identity = None
        state.lock_file_fd = fd
        state.lock_file_fd_identity = identity
        if isinstance(self._context, ThreadLocalFileContext):
            self._thread_pending.pop(id(state), None)
            self._thread_holds[id(state)] = state, current_thread()

    def _mark_descriptor_pending(self, fd: int, identity: tuple[int, int] | None = None) -> None:
        state: Final = self._hold_state()
        state.pending_lock_file_fd = fd
        state.pending_lock_file_fd_identity = identity
        if isinstance(self._context, ThreadLocalFileContext):
            self._thread_pending[id(state)] = state, current_thread()

    def _mark_descriptor_released(self) -> None:
        state: Final = self._hold_state()
        state.pending_lock_file_fd = None
        state.pending_lock_file_fd_identity = None
        state.lock_file_fd = None
        state.lock_file_fd_identity = None
        if isinstance(self._context, ThreadLocalFileContext):
            self._thread_pending.pop(id(state), None)
            self._thread_holds.pop(id(state), None)

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        # fork copies the lock in whatever state the parent's threads left it, so give the child an unheld one.
        self._transition_lock = RLock()
        if isinstance(self._context, ThreadLocalFileContext):
            self._thread_holds.clear()
            self._thread_pending.clear()
        self._context.owner_claim_paths = ()
        self._context.claim_root = None
        self._context.lock_file_fd = None
        self._context.lock_file_fd_token = None
        self._context.lock_file_fd_identity = None
        self._context.pending_lock_file_fd = None
        self._context.pending_lock_file_fd_identity = None
        self._context.lock_counter = 0
        self._context.releasing = None
        self._context.acquiring = None
        self._context.lock_file_key = None
        self._context.lock_file_registry = None

    def _descriptors_for_fork(self) -> tuple[tuple[int, tuple[int, int] | None], ...]:  # pragma: needs fork
        descriptors: list[tuple[int, tuple[int, int] | None]] = []
        if self._context.lock_file_fd is not None and self._context.lock_file_fd_token is None:
            descriptors.append((self._context.lock_file_fd, self._context.lock_file_fd_identity))
        if self._context.pending_lock_file_fd is not None:
            descriptors.append((self._context.pending_lock_file_fd, self._context.pending_lock_file_fd_identity))
        return tuple(descriptors)

    def _raise_if_would_deadlock(self, canonical: str, *, timeout: float, blocking: bool) -> None:
        """
        Fail fast when a *different* live instance already holds this path in the current deadlock scope.

        Only the first, indefinitely-blocking acquire can self-deadlock this way: waiting in the OS primitive would
        block on a lock this flow already owns. A finite timeout or ``blocking=False`` keeps the normal Timeout path.
        """
        would_block = self._context.lock_counter == 1 and not self.is_locked and timeout < 0 and blocking
        if would_block and _registry.held.get(self._registry_key(canonical)) not in {None, id(self)}:
            self._context.lock_counter -= 1
            msg = (
                f"Deadlock: lock '{self.lock_file}' is already held by a different {self._deadlock_holder_desc}. "
                f"Use is_singleton=True to enable reentrant locking across instances."
            )
            raise RuntimeError(msg)

    def _registry_key(self, canonical: str) -> Hashable:
        return canonical if (scope := self._deadlock_scope()) is None else (scope, canonical)

    @staticmethod
    def _deadlock_scope() -> Hashable | None:
        """
        Execution unit whose own hold would deadlock a new acquire.

        ``None`` scopes holders to the thread, which the thread-local registry already separates. Async locks
        override it with the running task, since one event loop thread runs many tasks and only a reacquire from
        the *same* task can self-deadlock.
        """
        return None

    def _poll_until_acquired(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        timeout: float,
        poll_interval: float,
        start_time: float,
    ) -> None:
        lock_id = id(self)
        lock_filename = self.lock_file
        attempt = 0
        self._context.acquiring = inspect.currentframe()
        try:
            while True:
                self._raise_if_inherited()
                if not self.is_locked:
                    self._try_break_expired_lock()
                    _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
                    self._acquire_with_fork_tracking()
                    self._raise_if_inherited()
                if self.is_locked:
                    _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    return
                if self._check_give_up(
                    blocking=blocking,
                    cancel_check=cancel_check,
                    timeout=timeout,
                    start_time=start_time,
                ):
                    raise Timeout(lock_filename, self._unprobeable_holder())
                attempt += 1
                delay = self._poll_delay(poll_interval, attempt)
                if timeout >= 0:
                    delay = min(delay, max(start_time + timeout - time.perf_counter(), 0.0))
                msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                _LOGGER.debug(msg, lock_id, lock_filename, delay)
                time.sleep(delay)
        finally:
            self._context.acquiring = None

    def _poll_delay(self, poll_interval: float, attempt: int) -> float:
        # A single-file lock retries on a fixed cadence. A backend that publishes several files per acquisition sets a
        # cap, and then contending processes back off across a jittered, exponentially widening window instead of
        # colliding on every poll; poll_interval stays the floor so a lone waiter is still responsive.
        if not self._poll_backoff_cap:
            return poll_interval
        # Cap the exponent before doubling: under heavy contention attempt reaches the thousands, and 2**attempt would
        # overflow the float multiply long before the window itself stops growing past the cap.
        window = min(
            self._poll_backoff_cap, poll_interval * 2 ** min(attempt, _MAX_BACKOFF_EXPONENT)
        )  # pragma: needs hard-link
        return max(poll_interval, secrets.randbelow(int(window * 1_000_000) + 1) / 1_000_000)  # pragma: needs hard-link

    def _reconcile_failed_acquire(self, canonical: str) -> None:
        # An acquire that raised while still holding the native lock (a hook that failed and whose rollback could not
        # release) must keep the registry entry so a later release can retry the OS unlock; otherwise roll the counter
        # back. is_locked was already reconciled by whichever release ran.
        if self.is_locked:
            self._commit_acquire(canonical)
        else:
            self._undo_acquire()

    def _invoke_on_acquired(self) -> None:
        # The wrapper runs in the backend executor for async locks, preserving the callback's documented thread.
        if self._on_acquired is None or self._context.lock_counter != 1:
            return
        try:
            self._on_acquired(cast("int", self._context.lock_file_fd))
        except BaseException as callback_error:  # arbitrary caller code; roll back on any failure
            callback_context = callback_error.__context__
            try:
                self._release_with_fork_tracking()
            except BaseException as release_error:  # ruff:ignore[blind-except]  # both errors surface via the group below
                _raise_body_and_release(callback_error, release_error)
            callback_error.__context__ = callback_context
            raise

    def _acquire_with_fork_tracking(self) -> None:
        with _fork_transition(self):
            try:
                self._acquire()
            except BaseException as acquisition_error:
                self._rollback_failed_acquire(acquisition_error)
                raise
            try:
                self._register_context_descriptor()
            except BaseException as registration_error:  # pragma: needs fork
                self._rollback_failed_registration(registration_error)
                raise
        if self.is_locked:
            self._invoke_on_acquired()

    def _rollback_failed_acquire(self, acquisition_error: BaseException) -> None:
        if not self.is_locked:
            return
        registration_error: BaseException | None = None
        tracking_error: BaseException | None = None
        try:
            self._register_context_descriptor()
        except BaseException as error:  # ruff:ignore[blind-except]  # preserve registration and acquisition failures
            registration_error = error
            try:
                # Rollback may fail too; retain the fd so a child can close it without another identity probe.
                self._register_unverified_context_descriptor()
            except BaseException as error:  # ruff:ignore[blind-except]  # pragma: no cover - allocation/control-flow during fallback
                tracking_error = error
        try:
            self._release_with_fork_tracking()
        except BaseException as rollback_error:  # ruff:ignore[blind-except]  # preserve rollback and acquisition failures
            _raise_cleanup_errors(
                "lock acquisition cleanup failed",
                acquisition_error,
                registration_error,
                tracking_error,
                rollback_error,
            )
        if registration_error is not None:  # pragma: needs fork
            _raise_cleanup_errors(
                "lock acquisition cleanup failed", acquisition_error, registration_error, tracking_error
            )

    def _rollback_failed_registration(self, registration_error: BaseException) -> None:  # pragma: needs fork
        tracking_error: BaseException | None = None
        try:
            # Rollback may fail too; retain the fd so a child can close it without another identity probe.
            self._register_unverified_context_descriptor()
        except BaseException as error:  # ruff:ignore[blind-except]  # pragma: no cover - allocation/control-flow during fallback
            tracking_error = error
        try:
            self._release_with_fork_tracking()
        except BaseException as rollback_error:  # ruff:ignore[blind-except]  # preserve rollback and registration failures
            _raise_cleanup_errors(
                "descriptor registration cleanup failed", registration_error, tracking_error, rollback_error
            )
        if tracking_error is not None:  # pragma: no cover - requires failed in-memory fallback
            _raise_cleanup_errors("descriptor registration cleanup failed", registration_error, tracking_error)

    def _release_with_fork_tracking(self) -> None:
        with _fork_transition(self):
            try:
                self._release()
            finally:
                self._unregister_released_descriptor()

    def _register_context_descriptor(self) -> None:
        if self._context.lock_file_fd is not None and self._context.lock_file_fd_token is None:
            self._context.lock_file_fd_token = _register_owned_descriptor(
                self._context.lock_file_fd,
                self._context.lock_file_fd_identity,
            )

    def _register_unverified_context_descriptor(self) -> None:
        # The rollback only reaches here still holding a descriptor it never managed to register.
        if self._context.lock_file_fd is not None and self._context.lock_file_fd_token is None:  # pragma: no branch
            self._context.lock_file_fd_token = _register_unverified_owned_descriptor(self._context.lock_file_fd)

    def _unregister_released_descriptor(self) -> None:
        if self._context.lock_file_fd is None:
            if (token := self._context.lock_file_fd_token) is not None:  # pragma: needs fork
                _unregister_owned_descriptor(token)
            self._context.lock_file_fd_token = None
            self._context.lock_file_fd_identity = None

    def _undo_acquire(self) -> None:
        """Roll back the counter after a failed acquire, dropping the registry entry once nothing holds the path."""
        self._context.lock_counter = max(0, self._context.lock_counter - 1)
        if self._context.lock_counter == 0:
            self._drop_registry_entry()

    def _commit_acquire(self, canonical: str) -> None:
        """Record this instance as the holder once the first acquire succeeds, so peers can detect the deadlock."""
        # Keyed on the registry rather than the counter: an on_acquired hook that acquires again bumps the counter
        # past 1 before this first commit runs, and must not leave the hold unregistered.
        if self._context.lock_file_registry is None:
            key = self._registry_key(canonical)
            # The holder scope is resolved once at commit so a later release from another flow drops the right entry.
            self._context.lock_file_key = key
            # A shared instance may be released from another thread, which must clear the entry in the acquirer's
            # registry: the releaser's own thread-local registry never held it.
            self._context.lock_file_registry = _registry.held
            _registry.held[key] = id(self)

    def _drop_registry_entry(self) -> None:
        """Forget the key owned by this hold without resolving a mutable path again."""
        # Ordered so a run an interrupt cuts short repeats safely: the entry goes only while it is still ours, and the
        # key outlives the registry it indexes, so a later commit sees no registry and records the hold afresh.
        key, registry = self._context.lock_file_key, self._context.lock_file_registry
        if registry is not None and registry.get(key) == id(self):
            del registry[key]
        self._context.lock_file_registry = None
        self._context.lock_file_key = None

    def _commit_release(self) -> None:
        """Record the lock as fully released: reset the recursion counter and drop the deadlock-registry entry."""
        self._context.lock_counter = 0
        self._drop_registry_entry()

    def _try_break_expired_lock(self) -> None:
        """Remove the lock file if its modification time exceeds the configured :attr:`lifetime`."""
        if (lifetime := self._context.lifetime) is None:
            return
        with contextlib.suppress(OSError):
            # lstat, not stat: an attacker with write access to the lock directory can replace a held
            # lock file with a symlink pointing at an old file, making stat() report the target's stale
            # mtime so a waiter breaks a live lock and two processes hold it at once. lstat reads the
            # symlink's own mtime, matching the O_NOFOLLOW reads elsewhere.
            st = os.lstat(self.lock_file)
            if time.time() - st.st_mtime < lifetime:
                return
            break_lock_file(self.lock_file, st.st_mtime, st.st_ino)

    def _check_give_up(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        timeout: float,
        start_time: float,
    ) -> bool:
        lock_id, lock_filename = id(self), self.lock_file
        if blocking is False:
            _LOGGER.debug("Failed to immediately acquire lock %s on %s", lock_id, lock_filename)
            return True
        if cancel_check is not None and cancel_check():
            _LOGGER.debug("Cancellation requested for lock %s on %s", lock_id, lock_filename)
            return True
        if 0 <= timeout < time.perf_counter() - start_time:
            _LOGGER.debug("Timeout on acquiring lock %s on %s", lock_id, lock_filename)
            return True
        return False

    def _unprobeable_holder(self) -> str | None:  # ruff:ignore[no-self-use]  # existence locks override it
        return None

    @abstractmethod
    def _acquire(self) -> None:
        """If the file lock could be acquired, self._context.lock_file_fd holds the file descriptor of the lock file."""
        raise NotImplementedError

    @abstractmethod
    def _release(self) -> None:
        """Releases the lock and sets self._context.lock_file_fd to None."""
        raise NotImplementedError


def _resolve_read_write_timeout(timeout: float, *, blocking: bool) -> float:
    """Validate a read-write lock's acquire ``timeout``: ``-1`` or a non-negative number, ``inf`` meaning ``-1``."""
    if (timeout := _resolve_timeout(timeout, blocking=blocking)) < 0 and timeout != -1 and blocking:
        msg = f"timeout must be a non-negative number or -1, not {timeout!r}"
        raise ValueError(msg)
    return timeout


def _resolve_timeout(timeout: float, *, blocking: bool) -> float:
    """Validate an acquire ``timeout``; every negative value waits without a limit."""
    if math.isnan(timeout := _seconds("timeout", timeout)) and blocking:
        msg = "timeout must be a number of seconds, not nan"
        raise ValueError(msg)
    # The fail-fast deadlock check only sees a negative timeout as unlimited, so spell inf that way.
    return -1 if timeout == math.inf else timeout


def _on_this_stack(frame: FrameType | None) -> bool:
    if frame is None:
        return False
    caller = inspect.currentframe()
    while caller is not None and caller is not frame:
        caller = caller.f_back
    return caller is not None


# acquire() returns this wrapper instead of self so entering the with-statement does not call __enter__ a second
# time; returning self would re-acquire the lock in BaseFileLock.__enter__ without a matching release (issue #37).
class AcquireReturnProxy(Generic[_LockT]):
    """A context-aware object that will release the lock file when exiting."""

    def __init__(self, lock: _LockT) -> None:
        self.lock: _LockT = lock

    def __enter__(self) -> _LockT:
        return self.lock

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Widen to the bound: ty narrows the else branch to the type variable minus BaseFileLock, not to the union rest.
        lock: Final = cast("BaseFileLock | ReadWriteLock | SoftReadWriteLock", self.lock)
        if isinstance(lock, BaseFileLock):
            lock._release_in_context(exc_value)  # ruff:ignore[private-member-access]  # forwards __exit__ to the owned lock's context release
        else:  # a reader/writer lock does not carry a context_error_policy
            lock.release()


@dataclass
class FileLockContext:
    """Holds the context for a ``BaseFileLock`` object."""

    # A separate class so ThreadLocalFileContext can make the whole context thread-local.

    lock_file: str
    timeout: float
    mode: int
    blocking: bool
    poll_interval: float

    #: The lock lifetime in seconds; ``None`` means the lock never expires.
    lifetime: float | None = None

    #: File descriptor from os.open for the lock file; not None while the lock is held.
    lock_file_fd: int | None = None

    #: Registry token for the descriptor owned by this thread's context.
    lock_file_fd_token: int | None = None

    #: Identity captured by a backend that already inspected the descriptor.
    lock_file_fd_identity: tuple[int, int] | None = None

    #: Descriptor opened by a backend but not yet committed as the held lock.
    pending_lock_file_fd: int | None = None

    #: Identity captured for a descriptor whose acquisition has not committed.
    pending_lock_file_fd_identity: tuple[int, int] | None = None

    #: Depth of nested acquisitions; the lock is released only when it returns to 0.
    lock_counter: int = 0

    #: Canonical registry key captured when the first physical acquisition commits.
    lock_file_key: Hashable | None = None

    #: The acquiring thread's deadlock registry holding :attr:`lock_file_key`.
    lock_file_registry: dict[Hashable, int] | None = None

    #: Claim pathnames this owner published, removed by name on release so no holder ever unlinks a peer's claim.
    owner_claim_paths: tuple[str, ...] = ()

    #: The frame of the :meth:`BaseFileLock.release` tearing this hold down, so a re-entrant call from a signal handler
    #: on the same thread returns instead of closing the descriptor a second time.
    releasing: FrameType | None = None

    #: The frame of an acquire still polling for this hold, so one a signal handler starts on this thread fails fast.
    acquiring: FrameType | None = None

    #: Canonical lock path resolved when an acquisition starts. A waiter polling a relative path must keep publishing
    #: into the directory it started in, even when another thread changes the working directory mid-wait.
    claim_root: str | None = None

    #: The lease claim of this hold, kept with it so a cross-thread release stops the right heartbeat.
    lease_claim: _LeaseClaim | None = None


class ThreadLocalFileContext(FileLockContext, local):
    """
    A thread local version of the ``FileLockContext`` class.

    Each thread's state lives in an ordinary :class:`FileLockContext` at :attr:`current`, which the lock also keeps, so
    ``release(force=True)`` and the finalizer can unlock a hold whose thread exited.
    """

    current: FileLockContext

    # threading.local runs __init__ again with these arguments the first time each thread touches the context.
    def __init__(  # ruff:ignore[too-many-arguments]  # mirrors the FileLockContext fields set at construction
        self,
        *,
        lock_file: str,
        timeout: float,
        mode: int,
        blocking: bool,
        poll_interval: float,
        lifetime: float | None = None,
    ) -> None:
        self.current = FileLockContext(
            lock_file=lock_file,
            timeout=timeout,
            mode=mode,
            blocking=blocking,
            poll_interval=poll_interval,
            lifetime=lifetime,
        )


def _delegate_to_current(name: str) -> property:
    return property(
        lambda context: getattr(context.current, name),
        lambda context, value: setattr(context.current, name, value),
    )


for _field in fields(FileLockContext):
    setattr(ThreadLocalFileContext, _field.name, _delegate_to_current(_field.name))


@dataclass(frozen=True)
class _OwnedDescriptor:
    fd: int
    creator_pid: int
    device: int | None
    inode: int | None


class _ForkState:
    def __init__(self) -> None:
        self.gate = Condition(RLock())
        self.registry_lock = RLock()
        self.parameter_models_lock = RLock()
        self.transitions: dict[int, dict[int, _ForkDescriptorOwner | None]] = {}
        self.active_transitions = 0
        self.admission_closed = False
        self.fork_owner_depths: dict[int, int] = {}
        self.pinned_objects: dict[int, list[tuple[_ForkResettable, ...]]] = {}
        self.pinned_classes: dict[int, list[tuple[_ForkResettableClass, ...]]] = {}
        self.provisional_descriptor_tokens: dict[int, list[tuple[int, ...]]] = {}
        self.pid = os.getpid()

    def reset_synchronization(self) -> None:  # pragma: forked child
        self.gate = Condition(RLock())
        self.registry_lock = RLock()
        self.parameter_models_lock = RLock()
        self.transitions = {}
        self.active_transitions = 0
        self.admission_closed = False
        self.fork_owner_depths = {}
        self.pinned_objects = {}
        self.pinned_classes = {}
        self.provisional_descriptor_tokens = {}


_FORK_OBJECTS: Final[WeakValueDictionary[int, _ForkResettable]] = WeakValueDictionary()
_FORK_CLASSES: Final[WeakValueDictionary[int, _ForkResettableClass]] = WeakValueDictionary()
_OWNED_DESCRIPTORS: Final[dict[int, _OwnedDescriptor]] = {}
_DESCRIPTOR_TOKENS: Final[count[int]] = count()
_TRANSITION_TOKENS: Final[count[int]] = count()
_FORK_STATE: Final = _ForkState()
_FORK_AUDIT_EVENTS: Final[frozenset[str]] = frozenset({"os.fork", "os.forkpty"})
# CPython before 3.12 guards sys.settrace with one process-wide flag that stays set while audit hooks run, so a thread
# switch inside a Python hook makes a concurrent settrace raise. Coverage and debuggers call it on every thread start.
_SETTRACE_SURVIVES_AUDIT_HOOKS: Final[bool] = sys.implementation.name != "cpython" or sys.version_info >= (3, 12)


def _register_fork_hooks() -> None:
    if _REGISTER_AT_FORK is None:
        return  # pragma: lacks fork
    if _SETTRACE_SURVIVES_AUDIT_HOOKS:  # pragma: needs fork  # pragma: needs settrace-safe-audit-hooks
        sys.addaudithook(_audit_fork_safety)
    _REGISTER_AT_FORK(  # pragma: needs fork
        before=_pin_fork_objects,
        after_in_parent=_resume_parent_after_fork,
        after_in_child=_reset_child_after_fork,
    )


@contextmanager
def _fork_transition(descriptor_owner: _ForkDescriptorOwner | None = None) -> Generator[None]:
    # No fork follows exit, and a daemon thread stopped inside this bookkeeping holds its lock forever.
    if not _HAS_REGISTER_AT_FORK or sys.is_finalizing():
        yield  # pragma: lacks fork
        return  # pragma: lacks fork
    _ensure_current_process()  # pragma: needs fork
    creator_pid = os.getpid()  # pragma: needs fork
    # A generator holding this transition can be finalized on another thread, which must close the entering thread's.
    thread_id: Final = get_ident()  # pragma: needs fork
    token: Final = _enter_fork_transition(descriptor_owner, thread_id)  # pragma: needs fork
    try:  # pragma: needs fork
        yield
    finally:
        if os.getpid() == creator_pid:  # pragma: needs fork
            _leave_fork_transition(token, thread_id)


def _register_fork_object(instance: _ForkResettable) -> None:
    if not _HAS_REGISTER_AT_FORK:
        return  # pragma: lacks fork
    with _fork_transition(), _FORK_STATE.registry_lock:  # pragma: needs fork
        _FORK_OBJECTS[id(instance)] = instance
        _refresh_owner_pins()


def _register_fork_class(cls: _ForkResettableClass) -> None:
    if not _HAS_REGISTER_AT_FORK:
        return  # pragma: lacks fork
    with _fork_transition(), _FORK_STATE.registry_lock:  # pragma: needs fork
        _FORK_CLASSES[id(cls)] = cls
        _refresh_owner_pins()


def _register_owned_descriptor(fd: int, identity: tuple[int, int] | None = None) -> int | None:
    if not _HAS_REGISTER_AT_FORK:
        return None  # pragma: lacks fork
    with _fork_transition():  # pragma: needs fork
        if identity is None:
            stat_result = os.fstat(fd)
            identity = stat_result.st_dev, stat_result.st_ino
        return _record_owned_descriptor(fd, identity)


def _register_unverified_owned_descriptor(fd: int) -> int | None:
    if not _HAS_REGISTER_AT_FORK:
        return None  # pragma: lacks fork
    with _fork_transition():  # pragma: needs fork
        return _record_owned_descriptor(fd, None)


def _record_owned_descriptor(fd: int, identity: tuple[int, int] | None) -> int:  # pragma: needs fork
    with _FORK_STATE.registry_lock:
        token = next(_DESCRIPTOR_TOKENS)
        _OWNED_DESCRIPTORS[token] = _OwnedDescriptor(
            fd=fd,
            creator_pid=os.getpid(),
            device=None if identity is None else identity[0],
            inode=None if identity is None else identity[1],
        )
    return token


def _unregister_owned_descriptor(token: int) -> None:  # pragma: needs fork
    # No fork follows exit, and a daemon thread stopped inside this bookkeeping holds its lock forever.
    if sys.is_finalizing():  # pragma: no cover  # only exit reaches it, after coverage stops measuring
        return
    with _fork_transition(), _FORK_STATE.registry_lock:
        _OWNED_DESCRIPTORS.pop(token, None)


def _pin_fork_objects() -> None:  # pragma: needs fork
    _ensure_current_process()
    thread_id = get_ident()
    with _FORK_STATE.gate:
        _FORK_STATE.fork_owner_depths[thread_id] = _FORK_STATE.fork_owner_depths.get(thread_id, 0) + 1
        _FORK_STATE.admission_closed = True
        while _FORK_STATE.active_transitions > len(_FORK_STATE.transitions.get(thread_id, ())):
            _FORK_STATE.gate.wait()
        transition_owners = tuple(_FORK_STATE.transitions.get(thread_id, {}).values())
    _verify_unverified_descriptors()
    owners = {id(owner): owner for owner in transition_owners if owner is not None}
    provisional_descriptor_tokens = tuple(
        starmap(
            _snapshot_descriptor_for_fork,
            (descriptor for owner in owners.values() for descriptor in owner._descriptors_for_fork()),  # ruff:ignore[private-member-access]  # snapshots each owner's own fork descriptors
        )
    )
    with _FORK_STATE.registry_lock:
        _FORK_STATE.provisional_descriptor_tokens.setdefault(thread_id, []).append(provisional_descriptor_tokens)
        _FORK_STATE.pinned_objects.setdefault(thread_id, []).append(tuple(_FORK_OBJECTS.values()))
        _FORK_STATE.pinned_classes.setdefault(thread_id, []).append(tuple(_FORK_CLASSES.values()))


def _resume_parent_after_fork() -> None:  # pragma: needs fork
    thread_id = get_ident()
    with _FORK_STATE.registry_lock:
        for token in _FORK_STATE.provisional_descriptor_tokens[thread_id].pop():
            _OWNED_DESCRIPTORS.pop(token, None)
        _FORK_STATE.pinned_objects[thread_id].pop()
        _FORK_STATE.pinned_classes[thread_id].pop()
        if not _FORK_STATE.provisional_descriptor_tokens[thread_id]:  # pragma: needs fork
            del _FORK_STATE.provisional_descriptor_tokens[thread_id]
            del _FORK_STATE.pinned_objects[thread_id]
            del _FORK_STATE.pinned_classes[thread_id]
    with _FORK_STATE.gate:
        if _FORK_STATE.fork_owner_depths[thread_id] == 1:
            del _FORK_STATE.fork_owner_depths[thread_id]
        else:  # pragma: no cover - earlier at-fork callbacks may deadlock first
            _FORK_STATE.fork_owner_depths[thread_id] -= 1
        _FORK_STATE.admission_closed = bool(_FORK_STATE.fork_owner_depths)
        _FORK_STATE.gate.notify_all()


def _ensure_current_process() -> None:
    _reset_child_after_fork()


def _reset_child_after_fork() -> None:  # pragma: forked child
    if (pid := os.getpid()) == _FORK_STATE.pid:
        return
    thread_id = get_ident()
    pinned_objects = _FORK_STATE.pinned_objects.get(thread_id, [()])[-1]
    pinned_classes = _FORK_STATE.pinned_classes.get(thread_id, [()])[-1]
    descriptors: list[_OwnedDescriptor] = []
    for token, descriptor in tuple(_OWNED_DESCRIPTORS.items()):
        if descriptor.creator_pid != pid:
            descriptors.append(_OWNED_DESCRIPTORS.pop(token))
    _FORK_STATE.reset_synchronization()
    _FORK_STATE.pid = pid
    _INIT_PARAMETER_MODELS.clear()
    _detach_child_state(descriptors, pinned_objects, pinned_classes)


def _enter_fork_transition(descriptor_owner: _ForkDescriptorOwner | None, thread_id: int) -> int:  # pragma: needs fork
    with _FORK_STATE.gate:
        while (
            _FORK_STATE.admission_closed
            and thread_id not in _FORK_STATE.fork_owner_depths
            and thread_id not in _FORK_STATE.transitions
        ):
            _FORK_STATE.gate.wait()  # pragma: no cover - exercised in isolated interpreter
        token = next(_TRANSITION_TOKENS)
        _FORK_STATE.transitions.setdefault(thread_id, {})[token] = descriptor_owner
        _FORK_STATE.active_transitions += 1
    return token


def _leave_fork_transition(token: int, thread_id: int) -> None:  # pragma: needs fork
    with _FORK_STATE.gate:
        owned: Final = _FORK_STATE.transitions[thread_id]
        del owned[token]
        if not owned:
            del _FORK_STATE.transitions[thread_id]
        _FORK_STATE.active_transitions -= 1
        _FORK_STATE.gate.notify_all()


def _verify_unverified_descriptors() -> None:  # pragma: needs fork
    with _FORK_STATE.registry_lock:
        for token, descriptor in tuple(_OWNED_DESCRIPTORS.items()):
            if descriptor.creator_pid != os.getpid() or descriptor.device is not None:
                continue
            try:
                stat_result = os.fstat(descriptor.fd)
            except OSError:
                continue
            _OWNED_DESCRIPTORS[token] = _OwnedDescriptor(
                fd=descriptor.fd,
                creator_pid=descriptor.creator_pid,
                device=stat_result.st_dev,
                inode=stat_result.st_ino,
            )


def _snapshot_descriptor_for_fork(fd: int, identity: tuple[int, int] | None) -> int:  # pragma: needs fork
    if identity is None:  # pragma: needs fork
        try:
            stat_result = os.fstat(fd)
        except OSError:
            pass
        else:
            identity = stat_result.st_dev, stat_result.st_ino
    return _record_owned_descriptor(fd, identity)


def _detach_child_state(  # pragma: forked child
    descriptors: list[_OwnedDescriptor],
    pinned_objects: tuple[_ForkResettable, ...],
    pinned_classes: tuple[_ForkResettableClass, ...],
) -> None:
    for descriptor in descriptors:
        if descriptor.device is None:
            continue
        try:
            stat_result = os.fstat(descriptor.fd)
        except OSError:
            continue
        if (stat_result.st_dev, stat_result.st_ino) != (descriptor.device, descriptor.inode):
            continue
        with contextlib.suppress(OSError):
            os.close(descriptor.fd)
    for instance in pinned_objects:
        instance._reset_after_fork_in_child()  # ruff:ignore[private-member-access]  # resets each pinned instance in the fork child
    for cls in pinned_classes:
        cls._reset_class_after_fork()
    _registry.held.clear()


def _refresh_owner_pins() -> None:  # pragma: needs fork
    with _FORK_STATE.gate:
        fork_owner_thread_ids = tuple(_FORK_STATE.fork_owner_depths)
    if get_ident() not in fork_owner_thread_ids:  # pragma: no cover - at-fork callbacks disable tracing
        return
    objects = tuple(_FORK_OBJECTS.values())
    classes = tuple(_FORK_CLASSES.values())
    for thread_id in fork_owner_thread_ids:
        if object_snapshots := _FORK_STATE.pinned_objects.get(thread_id):  # pragma: needs fork
            object_snapshots[:] = [objects] * len(object_snapshots)
            _FORK_STATE.pinned_classes[thread_id][:] = [classes] * len(object_snapshots)


# The defaults capture the module globals: the hook outlives them at interpreter shutdown, where CPython wipes the
# module dict to None before the final audit events fire.
def _audit_fork_safety(  # pragma: no cover - CPython disables tracing while Python audit hooks run
    event: str,
    _args: Unused,
    *,
    _fork_events: frozenset[str] = _FORK_AUDIT_EVENTS,
    _state: _ForkState = _FORK_STATE,
    _get_ident: Callable[[], int] = get_ident,
) -> None:
    if event in _fork_events:
        if _get_ident() in _state.transitions or _state.fork_owner_depths:
            msg = f"{event} is unsafe while filelock is changing descriptor ownership"
            raise RuntimeError(msg)
    elif event == "_posixsubprocess.fork_exec" and _get_ident() in _state.transitions:
        msg = "fork_exec is unsafe while filelock is changing descriptor ownership"
        raise RuntimeError(msg)


_register_fork_hooks()


__all__ = [
    "_UNSET_FILE_MODE",
    "AcquireReturnProxy",
    "BaseFileLock",
    "CloseErrorPolicy",
    "ContextErrorPolicy",
    "FileLockContext",
    "FileLockMeta",
    "LockOptions",
    "ThreadLocalFileContext",
    "_append_exception_context",
    "_canonical",
    "_check_timeout_max",
    "_delegate_to_current",
    "_ensure_current_process",
    "_fork_transition",
    "_grouped_errors",
    "_raise_body_and_release",
    "_raise_chained_errors",
    "_raise_cleanup_errors",
    "_raise_grouped_errors",
    "_register_fork_class",
    "_register_fork_object",
    "_register_owned_descriptor",
    "_resolve_poll_interval",
    "_resolve_read_write_timeout",
    "_resolve_timeout",
    "_seconds",
    "_timeout_behavior",
    "_unregister_owned_descriptor",
]
<|endoftext|>from __future__ import annotations

import os
import secrets
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from threading import Event, Thread, current_thread
from typing import TYPE_CHECKING, Final, Literal, cast
from weakref import WeakMethod

from ._api import _check_timeout_max, _seconds
from ._error import LeaseSettingsMismatch
from ._identity import owner_is_stale
from ._marker import MarkerSoftFileLock, OwnerMode, OwnerRecord, parse_marker
from ._soft import _read_lock_file
from ._util import break_lock_file, touch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ._api import LockOptions, _ExtraValue

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Unpack
    else:  # pragma: no cover (<py311)
        from typing_extensions import Unpack

#: ``evicted`` is a :class:`~filelock.SoftReadWriteLock` loss: a peer waited out the stale threshold and committed a
#: generation without this holder.
CompromiseReason = Literal["marker-missing", "owner-changed", "refresh-failed", "evicted"]

_RefreshOutcome = Literal["ok", "transient", "marker-missing", "owner-changed"]
_DEFAULT_LEASE_DURATION: Final[float] = 30.0


@dataclass(frozen=True)
class LeaseCompromise:
    """Why a held lease stopped being this process's to hold."""

    lock_file: str
    token: str
    reason: CompromiseReason
    error: OSError | None = None


@dataclass(frozen=True)
class _Heartbeat:
    """A running heartbeat and the event that stops it, which only ever exist together."""

    thread: Thread
    stop: Event


@dataclass
class _LeaseClaim:
    """The state of one claim: its token, its heartbeat, and how that claim was lost."""

    token: str | None = None
    compromise: LeaseCompromise | None = None
    heartbeat: _Heartbeat | None = None


class SoftFileLease(MarkerSoftFileLock):
    """
    Existence lock whose claim expires, so a peer may take it while the previous holder still runs.

    A lease trades mutual exclusion for progress. The holder publishes a claim and refreshes it every
    ``heartbeat_interval`` seconds; a contender takes the marker once it has seen no refresh for ``lease_duration``
    seconds of its own monotonic clock, so clock skew between hosts cannot expire a live claim. Nothing stops the
    expired holder: it keeps running, and it keeps using whatever the lock protects. Treat the lease as a hint about
    who *should* be working, not as a guarantee that only one worker is.

    To make a protected resource reject a superseded holder, that resource must be linearizable and must fence on a
    monotonic generation it controls. :attr:`token` names a claim; it does not fence one. Where overlap is unacceptable,
    use :class:`StrictSoftFileLock <filelock.StrictSoftFileLock>` instead.

    Every contender for a path must agree on ``lease_duration``. A contender that finds a claim published under a
    different duration raises :class:`LeaseSettingsMismatch <filelock.LeaseSettingsMismatch>` rather than apply its own
    expiry to a peer that never agreed to it.

    Expiry reclaims less on Windows, which refuses to rename or delete a file another process holds open. A peer there
    takes an expired claim only once the previous holder's process exits and its handle closes; a holder that lives on
    but stops refreshing keeps the marker. Unix reclaims the marker either way.

    ``on_compromise`` fires from the heartbeat thread when a refresh fails, or when the marker vanishes or names another
    owner. The holder should stop touching the protected resource when it runs. Because it runs on that thread, a
    ``release()`` inside it only takes effect when the lease was built with ``thread_local=False``; the default
    thread-local context hides the claim from every thread but the one that acquired it, so the release does nothing.
    Signal the acquiring thread instead when the context stays thread-local.

    .. versionadded:: 3.30.0

    """

    _owner_mode: OwnerMode = "lease"

    #: lease_duration replaces the legacy age-based lifetime, so accepting both would give one lock two expiry clocks.
    _lifetime_supported: bool = False
    _lifetime_unsupported_reason: str = "lease_duration sets when a lease expires"

    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        *,
        lease_duration: float = _DEFAULT_LEASE_DURATION,
        heartbeat_interval: float | None = None,
        on_compromise: Callable[[LeaseCompromise], None] | None = None,
        **kwargs: Unpack[LockOptions],
    ) -> None:
        """
        Create a lease.

        :param lease_duration: seconds a contender must see the marker go unrefreshed before it may take the claim.
            Every contender for the path must pass the same value.
        :param heartbeat_interval: seconds between refreshes. Defaults to a third of ``lease_duration``, leaving room
            for two missed refreshes before a peer may take the claim. Must be shorter than ``lease_duration`` and at
            most :data:`threading.TIMEOUT_MAX`.
        :param on_compromise: called from the heartbeat thread with a :class:`LeaseCompromise` when the claim is lost.
        :param kwargs: every other :class:`BaseFileLock <filelock.BaseFileLock>` option, ``timeout`` and ``mode`` among
            them. The metaclass passes them all by keyword, and taking them here lets
            :class:`AsyncSoftFileLease <filelock.AsyncSoftFileLease>` add the async plumbing a fixed signature would
            hide.

        """
        # A plain float: the marker records its repr, which NumPy's float64 spells in a form no parser reads.
        if not isfinite(lease_duration := _seconds("lease_duration", lease_duration)) or lease_duration <= 0:
            msg = f"lease_duration must be positive and finite, got {lease_duration!r}"
            raise ValueError(msg)
        heartbeat_interval = (
            lease_duration / 3 if heartbeat_interval is None else _seconds("heartbeat_interval", heartbeat_interval)
        )
        if not 0 < heartbeat_interval < lease_duration:
            msg = f"heartbeat_interval must be positive and below lease_duration, got {heartbeat_interval!r}"
            raise ValueError(msg)
        _check_timeout_max("heartbeat_interval", heartbeat_interval)
        super().__init__(lock_file, **kwargs)
        self._lease_duration = lease_duration
        self._heartbeat_interval = heartbeat_interval
        self._on_compromise = on_compromise
        # Two threads lazily creating a shared claim could each get one, and release() would miss the heartbeat's.
        if not self.is_thread_local():
            self._context.lease_claim = _LeaseClaim()

    def _singleton_extra_mismatches(self, kwargs: Mapping[str, _ExtraValue], /) -> dict[str, tuple[str, str]]:
        mismatches = super()._singleton_extra_mismatches(kwargs)
        lease_duration: Final = _seconds(
            "lease_duration", cast("float", kwargs.get("lease_duration", _DEFAULT_LEASE_DURATION))
        )
        if lease_duration != self._lease_duration:
            mismatches["lease_duration"] = (str(lease_duration), str(self._lease_duration))
        # An omitted heartbeat_interval resolves as in __init__; a differing duration is already reported above.
        heartbeat_interval: Final = (
            self._lease_duration / 3
            if (passed := kwargs.get("heartbeat_interval")) is None
            else _seconds("heartbeat_interval", cast("float", passed))
        )
        if heartbeat_interval != self._heartbeat_interval:
            mismatches["heartbeat_interval"] = (str(passed), str(self._heartbeat_interval))
        # A callback compares by identity, as on_acquired does: two equal callables can close over different state.
        if (on_compromise := kwargs.get("on_compromise")) is not self._on_compromise:
            mismatches["on_compromise"] = (str(on_compromise), str(self._on_compromise))
        return mismatches

    @property
    def _claim(self) -> _LeaseClaim:
        # Per hold, so another thread's failed acquire cannot stop the holder's heartbeat.
        state: Final = self._hold_state()
        if (claim := state.lease_claim) is None:
            claim = state.lease_claim = _LeaseClaim()
        return claim

    @property
    def lease_duration(self) -> float:
        """The staleness in seconds after which a contender may take this claim."""
        return self._lease_duration

    @property
    def token(self) -> str | None:
        """
        The token naming the claim this process published.

        :returns: the token while the lease is held, ``None`` otherwise. It identifies a claim; it does not fence one.

        """
        return self._claim.token

    @property
    def compromise(self) -> LeaseCompromise | None:
        """
        The loss of claim the heartbeat observed.

        The value outlives ``release()``, so a holder can check after leaving the ``with`` block whether its work may
        have overlapped a successor; the next acquisition resets it, as :attr:`SoftReadWriteLock.compromise
        <filelock.SoftReadWriteLock.compromise>` does.

        :returns: the :class:`LeaseCompromise` of the latest claim, or ``None`` while that claim holds or held to its
            release

        """
        return self._claim.compromise

    def _acquire(self) -> None:
        claim = self._claim
        self._stop_heartbeat()  # no earlier claim's heartbeat outlives the acquisition of the next one
        # The published record reads the token, so it exists before the marker is written and goes back on failure.
        claim.token = token = secrets.token_hex(16)
        claim.compromise = None
        try:
            super()._acquire()
        except BaseException:
            claim.token = None
            raise
        # The context is thread-local by default, so the heartbeat thread cannot read the descriptor this one just
        # published, nor the claim this one owns. Hand it the fd, the inode it verified and the claim instead.
        if (fd := self._context.lock_file_fd) is not None and (
            identity := self._context.lock_file_fd_identity
        ) is not None:
            self._start_heartbeat(claim, fd, identity, token)
        else:
            claim.token = None

    def _release(self) -> None:
        self._stop_heartbeat()
        self._claim.token = None
        super()._release()

    def _published_record(self) -> OwnerRecord:
        return super()._published_record()._replace(token=self._claim.token, lease_duration=self._lease_duration)

    def _try_break_stale_lock(self) -> None:
        if (peer := self._read_peer()) is None:
            # Not a readable protocol 2 lease record: a partial write, a foreign or legacy protocol 1 marker, or the
            # strict sentinel. The base self-heal evicts a genuinely malformed marker once it ages past the grace
            # window and leaves a legitimate legacy or strict holder in place, so a corrupt marker no longer wedges
            # every lease contender until its own timeout.
            super()._try_break_stale_lock()
            return
        owner, mtime, ino = peer
        # Only a peer that published a lease agreed to be superseded by age; an exclusive owner is reclaimed only once
        # it is provably dead, and a contract this version does not know never is. Raise the mismatch outside the read
        # so the suppression cannot swallow it.
        if owner.mode == "lease" and owner.lease_duration != self._lease_duration:
            msg = (
                f"{self.lock_file} holds a lease of {owner.lease_duration!r}s but this contender configured "
                f"{self._lease_duration!r}s; every contender for a path must agree on lease_duration"
            )
            raise LeaseSettingsMismatch(msg)
        # A break can fail for reasons a contender must ride out rather than raise on: a peer broke the marker first,
        # or Windows refuses to rename a file whose holder still has it open. Poll again instead.
        with suppress(OSError):
            # A dead or recycled owner is reclaimed at once, whether it held a lease or an exclusive marker; a live
            # lease owner whose marker stops being refreshed is superseded on the schedule every contender agreed to.
            if owner.mode != "unknown" and owner_is_stale(owner.pid, owner.hostname, owner.start):
                break_lock_file(self.lock_file, mtime, ino)
                return
            if owner.mode == "lease" and self._marker_unchanged_for(mtime, ino) >= self._lease_duration:
                break_lock_file(self.lock_file, mtime, ino)

    def _read_peer(self) -> tuple[OwnerRecord, float, int] | None:
        with suppress(OSError, ValueError):
            content, mtime, ino = _read_lock_file(self.lock_file)
            if (owner := parse_marker(content)) is not None:
                return owner, mtime, ino
        return None

    def _start_heartbeat(self, claim: _LeaseClaim, fd: int, identity: tuple[int, int], token: str) -> None:
        # The thread watches the event it was handed rather than whatever the claim names later: a heartbeat that
        # outlives its join timeout would otherwise adopt the next acquisition's event and never stop.
        stop: Final = Event()
        loop: Final = _RefreshLoop(
            # A strong reference from a running thread would keep a dropped lease alive, so __del__ would never release
            # it and its marker would outlive the process. The callback stops a heartbeat that __del__ could not reach.
            report=WeakMethod(self._report_compromise, lambda _: stop.set()),
            lock_file=self.lock_file,
            claim=claim,
            fd=fd,
            identity=identity,
            token=token,
            stop=stop,
            interval=self._heartbeat_interval,
            duration=self._lease_duration,
        )
        thread: Final = Thread(target=loop.run, name=f"filelock-lease-{os.getpid()}", daemon=True)
        # Record the heartbeat before starting the thread so a release racing this acquire on a shared,
        # non-thread-local claim always sees it and sets the stop event; the thread then exits at its first wait
        # instead of outliving the release. A start that raises leaves the unstarted thread for _stop_heartbeat.
        claim.heartbeat = _Heartbeat(thread, stop)
        thread.start()

    def _stop_heartbeat(self) -> None:
        claim = self._claim
        if (heartbeat := claim.heartbeat) is None:
            return
        heartbeat.stop.set()
        claim.heartbeat = None
        # An unstarted thread has nothing to join, on_compromise may release from the heartbeat thread itself, and
        # during finalization a daemon thread is frozen, so joining it raises PythonFinalizationError (3.14+).
        if heartbeat.thread.ident is not None and heartbeat.thread is not current_thread() and not sys.is_finalizing():
            heartbeat.thread.join(timeout=self._heartbeat_interval)

    def _report_compromise(
        self,
        claim: _LeaseClaim,
        reason: CompromiseReason,
        error: OSError | None,
        token: str,
    ) -> None:
        # Record it on the claim this heartbeat serves, not on self._claim: a thread-local claim read from the
        # heartbeat thread is a different, empty one, so the holder would never see the loss it is being told about.
        # The token is the one this thread published, not claim.token, which a release may already have cleared.
        claim.compromise = LeaseCompromise(lock_file=self.lock_file, token=token, reason=reason, error=error)
        if self._on_compromise is not None:
            self._on_compromise(claim.compromise)


@dataclass(frozen=True)
class _RefreshLoop:
    """What a heartbeat thread needs to refresh one claim, holding the lease itself only weakly."""

    report: WeakMethod[Callable[[_LeaseClaim, CompromiseReason, OSError | None, str], None]]
    lock_file: str
    claim: _LeaseClaim
    fd: int
    identity: tuple[int, int]
    token: str
    stop: Event
    interval: float
    duration: float

    def run(self) -> None:
        # A transient error (ESTALE / EIO on the NFS-style filesystems a lease targets) is not a loss. Like restic, call
        # the claim unrefreshable once failures run long enough that a contender could take it before the next success.
        last_success, missing = time.monotonic(), False
        while not self.stop.wait(self.interval):
            outcome, error = self._refresh()
            if outcome == "ok":
                last_success, missing = time.monotonic(), False
                continue
            # A peer's stale break moves a live marker aside for two syscalls, so only a second miss in a row is a loss.
            if outcome == "marker-missing" and not missing:
                missing = True
                continue
            if outcome == "transient" and time.monotonic() - last_success < self.duration - self.interval:
                continue
            if (report := self.report()) is not None:
                report(self.claim, "refresh-failed" if outcome == "transient" else outcome, error, self.token)
            return

    def _refresh(self) -> tuple[_RefreshOutcome, OSError | None]:
        try:
            st = os.lstat(self.lock_file)
        except FileNotFoundError as error:
            return "marker-missing", error
        except OSError as error:
            return "transient", error
        # A peer that took the expired claim replaced the marker, so the pathname now names its inode, not ours.
        if (st.st_dev, st.st_ino) != self.identity:
            return "owner-changed", None
        try:
            touch(self.lock_file, fd=self.fd)
        except OSError as error:
            return "transient", error
        return "ok", None


__all__ = [
    "CompromiseReason",
    "LeaseCompromise",
    "SoftFileLease",
    "_LeaseClaim",
]
<|endoftext|>from __future__ import annotations

import contextlib
import errno
import functools
import os
import secrets
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from errno import EACCES, EEXIST, ENOENT, ENOSYS, EPERM, ESTALE, EXDEV
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from ._api import BaseFileLock, ThreadLocalFileContext, _canonical, _raise_cleanup_errors
from ._error import SoftFileLockProtocolError
from ._identity import host_name, owner_is_stale, process_start_token
from ._marker import parse_marker
from ._soft import MALFORMED_LOCK_AGE_THRESHOLD, _parse_lock_holder
from ._soft_protocol import STRICT_SOFT_SENTINEL_RECORD
from ._util import ensure_directory_exists, write_all

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

StrictSoftFileClaimState = Literal["intent", "held"]
_OccupantState = Literal["sentinel", "held", "exited", "damaged", "unparsable"]

_CLAIM_STATES: Final[frozenset[str]] = frozenset({"intent", "held"})
_COORDINATION_SUFFIX: Final[str] = ".filelock"
_CLAIM_DIRECTORY_NAME: Final[str] = "claims"
_CLAIM_MAGIC: Final[str] = "filelock-strict-v1"
_CLAIM_RECORD_LIMIT: Final[int] = 1024
_CLAIM_NAME_PART_COUNT: Final[int] = 3
_TOKEN_HEX_LENGTH: Final[int] = 32
_PRIVATE_RECORD_MARKER: Final[str] = ".private-v1-"
_PRIVATE_RECORD_SUFFIX: Final[str] = ".tmp"
_PRIVATE_RECORD_RANDOM_HEX_LENGTH: Final[int] = 32
_PRIVATE_RECORD_GRACE: Final[float] = 2.0
_UNLINK_MAX_RETRIES: Final[int] = 10
#: How long a scan waits out a claim held in Windows' delete-pending state before treating it as unreadable.
_CLAIM_READ_GRACE: Final[float] = 0.5
_CLAIM_READ_RETRY: Final[float] = 0.002
#: Rides out a peer replacing the lock path mid-inspection; a foreign node or unreadable record outlasting it is damage.
_SENTINEL_READ_GRACE: Final[float] = 0.5
_NODE_DESCRIPTIONS: Final[dict[int, str]] = {stat.S_IFDIR: "a directory", stat.S_IFLNK: "a symlink"}
#: Windows opens descriptors in text mode by default, which rewrites newlines and truncates a record at a control byte.
#: The claim and sentinel records are exact binary, so every record descriptor must be binary; POSIX ignores the flag.
_O_BINARY: Final[int] = getattr(os, "O_BINARY", 0)
_LEGACY_SENTINEL: Final[bytes] = STRICT_SOFT_SENTINEL_RECORD.encode()
_WINDOWS_HARD_LINK_UNSUPPORTED: Final[frozenset[int]] = frozenset({1, 17, 50})

# Termux/Android CPython ships without os.link (bionic long had only linkat), so the strict backend's whole hard-link
# mechanism is absent there. Probe once and gate every os.link reference on it, so importing filelock still works and
# only an actual StrictSoftFileLock acquire reports the missing capability.
_HAS_LINK: Final[bool] = hasattr(os, "link")

# Probe dir_fd capability once at import. A per-call ``os.unlink in os.supports_dir_fd`` check flips to False the moment
# a test mocks os.unlink, silently diverting the code to a different branch than the one under test.
_OPEN_SUPPORTS_DIR_FD: Final[bool] = os.open in os.supports_dir_fd
_UNLINK_SUPPORTS_DIR_FD: Final[bool] = os.unlink in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD: Final[bool] = os.stat in os.supports_dir_fd
_LINK_SUPPORTS_DIR_FD: Final[bool] = _HAS_LINK and os.link in os.supports_dir_fd


def _probe_link_follow_symlinks() -> bool:
    # os.supports_follow_symlinks lists os.link on PyPy, but its linkat then rejects follow_symlinks=False with EINVAL,
    # and Windows raises NotImplementedError for the option outright. Link a throwaway file for real so the answer
    # reflects the runtime rather than its advertisement, and treat any failure as "not honored": the option only
    # hardens a source this process created with O_EXCL, so skipping it is safe, and a real environment fault surfaces
    # when the actual link runs.
    if not _HAS_LINK:
        return False
    try:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "probe-source")
            source.touch()
            os.link(source, Path(directory, "probe-link"), follow_symlinks=False)
    except (OSError, NotImplementedError, ValueError):
        return False
    return True


_LINK_HONORS_FOLLOW_SYMLINKS: Final[bool] = _probe_link_follow_symlinks()


def _probe_hard_link_unsupported_errnos() -> frozenset[int]:
    # GraalPy's errno omits ENOTSUP, so importing the name outright breaks every runtime that ships without it. ENOTSUP
    # wins wherever it exists, leaving every runtime that names it unchanged. EOPNOTSUPP only stands in for the ones
    # that do not, and it approximates rather than matches: the two codes agree on Linux but differ on macOS/BSD and on
    # Windows. Where neither exists a runtime that cannot name "operation not supported" cannot raise it either, and
    # ENOSYS/EXDEV still classify the link failures it can raise.
    not_supported = getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", None))
    return frozenset({ENOSYS, EXDEV} if not_supported is None else {ENOSYS, EXDEV, not_supported})


_HARD_LINK_UNSUPPORTED_ERRNOS: Final[frozenset[int]] = _probe_hard_link_unsupported_errnos()


class StrictSoftFileLock(BaseFileLock):
    """Portable fail-closed lock based on immutable owner claims."""

    _preserve_lock_file_supported: bool = True
    _on_acquired_supported: bool = False
    #: Age cannot clear a strict claim: expiring one on a clock is the overlap the fail-closed contract exists to rule
    #: out, so only force_break() removes it.
    _lifetime_supported: bool = False
    _lifetime_unsupported_reason: str = "a strict claim is never broken by age, only by force_break()"
    #: Contending processes each publish and rescan several files, so jitter their retries. Seconds. Only a
    #: poll_interval below this cap jitters; the default equals it and keeps a fixed cadence.
    _poll_backoff_cap: float = 0.05
    _occupant_seen: tuple[tuple[float, int, int], float] | None = None

    def _acquire(self) -> None:
        # Finish a doorway whose claim cleanup failed before publishing, so this owner never contends with leftovers.
        self._discard_pending_doorways(exited_only=True)
        # Resolve once per acquisition, not per poll: a waiter on a relative path must keep publishing into the
        # directory it started waiting in even when another thread changes the working directory mid-wait.
        if (claim_root := self._context.claim_root) is None:
            claim_root = self._context.claim_root = _canonical(self.lock_file)
        lock_path = Path(claim_root)
        coordination_directory = Path(f"{lock_path}{_COORDINATION_SUFFIX}")
        claim_directory = coordination_directory / _CLAIM_DIRECTORY_NAME
        ensure_directory_exists(os.fspath(lock_path))
        _ensure_protocol_directory(self.lock_file, coordination_directory)
        _ensure_protocol_directory(self.lock_file, claim_directory)
        if (
            sentinel_fd := _open_or_create_sentinel(
                self.lock_file, lock_path, self._open_mode(), self._occupant_unchanged_for
            )
        ) is None:
            return
        try:
            sentinel_identity = _file_identity(os.fstat(sentinel_fd))
        except BaseException as inspection_error:  # preserve inspection and descriptor cleanup errors
            try:
                os.close(sentinel_fd)
            except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve inspection and descriptor cleanup errors
                _raise_cleanup_errors("strict sentinel inspection cleanup failed", inspection_error, close_error)
            raise
        self._mark_descriptor_pending(sentinel_fd, sentinel_identity)
        try:
            won = self._attempt_doorway(claim_directory)
        except BaseException:
            self._discard_doorway()
            raise
        if won:
            self._mark_descriptor_owned(sentinel_fd, sentinel_identity)
        else:
            self._discard_doorway()

    def _discard_pending_doorways(self, *, exited_only: bool) -> None:
        if self._context.owner_claim_paths and not self.is_locked:
            self._discard_doorway()
        if not isinstance(context := self._context, ThreadLocalFileContext):
            return
        # Only this registry reaches an exited thread's thread-local doorway; a live thread finishes its own.
        own: Final = context.current
        with self._transition_lock:
            for pending, thread in self._thread_pending.copy().values():
                if exited_only and thread.is_alive():
                    continue
                context.current = pending
                try:
                    self._discard_doorway()
                finally:
                    context.current = own

    def _occupant_unchanged_for(self, occupant: os.stat_result) -> float:
        # SoftFileLock's marker rule: the mtime is only tested for change, timed on this process's monotonic clock.
        key, now = (occupant.st_mtime, occupant.st_ino, occupant.st_size), time.monotonic()
        if (seen := self._occupant_seen) is None or seen[0] != key:
            self._occupant_seen = key, now
            return 0.0
        return now - seen[1]

    def _attempt_doorway(self, claim_directory: Path) -> bool:
        if _read_existing_claims(self.lock_file, claim_directory):
            return False

        token = secrets.token_hex(_TOKEN_HEX_LENGTH // 2)
        intent_name = _claim_name("intent", token)
        intent_path = str(claim_directory / intent_name)
        # Record each claim before its link: once the link may have committed, any later failure must leave the name for
        # _discard_doorway. The token makes the name unique, so removing it when the link never happened is a no-op.
        self._context.owner_claim_paths = (intent_path,)
        try:
            publication_cleanup_error = _publish_record(intent_path, _claim_record(token), self._open_mode())
        except _PrivateRecordReclaimedError:
            return False
        except (NotImplementedError, OSError) as error:
            _raise_if_hard_links_unsupported(self.lock_file, error)
            if isinstance(error, OSError) and error.errno == EEXIST:
                # Another file holds the name, so it is not this owner's to remove.
                self._context.owner_claim_paths = ()
                return False
            raise
        if publication_cleanup_error is not None:  # pragma: needs dir-fd
            raise publication_cleanup_error

        claims = _read_existing_claims(self.lock_file, claim_directory)
        if (
            not claims
            or any(claim.state == "held" for claim in claims)
            or min(claim.name for claim in claims) != intent_name
        ):
            return False

        held_name = _claim_name("held", token)
        held_path = str(claim_directory / held_name)
        self._context.owner_claim_paths = (held_path, intent_path)
        try:
            link_cleanup_error = _link_no_replace(claim_directory, intent_name, held_name)
        except (NotImplementedError, OSError) as error:
            _raise_if_hard_links_unsupported(self.lock_file, error)
            if isinstance(error, OSError) and error.errno == EEXIST:
                self._context.owner_claim_paths = (intent_path,)
            raise
        if link_cleanup_error is not None:  # pragma: needs dir-fd
            raise link_cleanup_error

        claims = _read_existing_claims(self.lock_file, claim_directory)
        # Keep the intent claim for the whole hold rather than unlinking it now. The intent has existed, unchanged,
        # since this owner published it, so a contender's os.scandir is guaranteed to return it (POSIX only leaves the
        # visibility of entries created or removed *during* a scan unspecified). The freshly linked held claim carries
        # no such guarantee: a scan that races its creation can miss it. Were the intent removed here, that scan could
        # observe neither claim and let a larger-token contender win over this owner. The stable intent is the witness
        # that keeps the phase-five min-token decision computed over the true set. Release unlinks both.
        return {intent_name, held_name}.issubset(claim.name for claim in claims) and min(
            _claim_token_key(claim.name) for claim in claims
        ) == f"v1-{token}.claim"

    @property
    def claims(self) -> tuple[StrictSoftFileClaim, ...]:
        """Published claims that block acquisition."""
        # An inspection writes nothing, so it trusts no record's age rather than probe the filesystem clock.
        return _read_existing_claims(self.lock_file, self._claim_directory, probe_clock=False)

    @property
    def lock_path_occupant(self) -> str | None:
        """
        What sits at the lock path in place of the strict sentinel, or ``None`` when the sentinel or nothing is there.

        A soft-lock owner's marker blocks acquisition without appearing in :attr:`~filelock.StrictSoftFileLock.claims`;
        this describes it, naming the PID and host when the marker records them. A directory, a symlink, a marker whose
        owner on this host has exited, or a file it cannot parse that stays unchanged for two seconds makes acquisition
        raise :class:`~filelock.SoftFileLockProtocolError` instead and needs an operator to remove it.
        """
        try:
            fd, _, occupant = _inspect_lock_path(Path(_canonical(self.lock_file)))
        except FileNotFoundError:
            return None
        if fd is None:
            return occupant
        os.close(fd)
        return None

    def force_break(self, claim_name: str) -> None:
        """Remove one named claim, allowing overlap if its owner still holds the protected resource."""
        _validate_force_break_name(claim_name)
        _require_exact_name(self._claim_directory, claim_name)
        if (
            cleanup_error := _unlink_in_directory(self._claim_directory, claim_name)
        ) is not None:  # pragma: needs dir-fd
            raise cleanup_error

    def __del__(self) -> None:
        super().__del__()
        if vars(self).get("_creator_pid") != os.getpid():
            return  # pragma: forked child
        # release() skips a doorway whose cleanup failed, as it is no hold, so a dropped instance would strand claims.
        with contextlib.suppress(Exception):
            # GraalPy defers finalizers to the host collector, so its tests cannot drive this cleanup.
            self._discard_pending_doorways(exited_only=False)  # pragma: needs collected-finalization

    def _reconcile_failed_acquire(self, canonical: str) -> None:
        # The acquisition is over, so the next one resolves the working directory again rather than reuse this one's.
        self._context.claim_root = None
        super()._reconcile_failed_acquire(canonical)

    def _release(self) -> None:
        fd = cast("int", self._context.lock_file_fd)
        self._context.claim_root = None
        remaining, errors = _unlink_owner_paths(self._context.owner_claim_paths)
        self._context.owner_claim_paths = tuple(remaining)
        if remaining:
            _raise_recorded_errors("strict claim release failed", errors)
        self._mark_descriptor_released()
        try:
            self._close_released_fd(fd, default_suppresses=False)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve claim and sentinel cleanup errors
            errors.append(close_error)
        if errors:
            _raise_recorded_errors("strict release cleanup failed", errors)

    def _discard_doorway(self) -> None:
        remaining, errors = _unlink_owner_paths(self._context.owner_claim_paths)
        self._context.owner_claim_paths = tuple(remaining)
        if remaining:
            # Stay pending, not owned: is_locked reports only a held claim that passed the final rescan. The next
            # acquire retries this cleanup.
            _raise_recorded_errors("strict doorway claim cleanup failed", errors)
        fd = cast("int", self._context.pending_lock_file_fd)
        self._mark_descriptor_released()
        try:
            self._close_released_fd(fd, default_suppresses=False)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve claim and sentinel cleanup errors
            errors.append(close_error)
        if errors:
            _raise_recorded_errors("strict doorway cleanup failed", errors)

    @property
    def _claim_directory(self) -> Path:
        if self._context.owner_claim_paths:
            return Path(self._context.owner_claim_paths[0]).parent
        return Path(f"{_canonical(self.lock_file)}{_COORDINATION_SUFFIX}") / _CLAIM_DIRECTORY_NAME


@dataclass(frozen=True)
class StrictSoftFileClaim:
    """One parsed strict soft-lock claim."""

    name: str
    state: StrictSoftFileClaimState
    token: str
    pid: int
    hostname: str
    #: The owner's process start token, or ``None`` when the platform exposes no proven start time. A strict lock never
    #: reclaims a claim on its own, so this identifies the owner for tooling rather than driving any automatic break.
    start: int | None = None


class _PrivateRecordReclaimedError(Exception):
    pass


def _open_or_create_sentinel(
    lock_file: str, path: Path, mode: int, unchanged_for: Callable[[os.stat_result], float]
) -> int | None:
    try:
        return _open_sentinel(lock_file, path, unchanged_for)
    except FileNotFoundError:
        pass

    _reclaim_sentinel_private_records(path)
    try:
        publication_cleanup_error = _publish_record(os.fspath(path), _LEGACY_SENTINEL, mode)
    except _PrivateRecordReclaimedError:
        return None
    except (NotImplementedError, OSError) as error:
        _raise_if_hard_links_unsupported(lock_file, error)
        if not isinstance(error, OSError) or error.errno != EEXIST:
            raise
    else:
        if publication_cleanup_error is not None:  # pragma: needs dir-fd
            raise publication_cleanup_error
    try:
        return _open_sentinel(lock_file, path, unchanged_for)
    except FileNotFoundError:
        return None


def _open_sentinel(lock_file: str, path: Path, unchanged_for: Callable[[os.stat_result], float]) -> int | None:
    # A live soft-lock owner blocks like any holder. Anything else raises: claims cannot show it and force_break()
    # cannot remove it, so waiting would wedge every strict contender with no diagnostic. A file no lock parses may be a
    # soft lock between its create and its write, so it raises only once unchanged as long as a soft contender waits.
    deadline: float | None = None
    while True:
        fd, state, occupant = _inspect_lock_path(path)
        if fd is not None or state == "held":
            return fd
        reason = f"expected the strict sentinel at the lock path, found {occupant}"
        if state == "unparsable":
            if unchanged_for(path.lstat()) < MALFORMED_LOCK_AGE_THRESHOLD:
                return None
            raise SoftFileLockProtocolError(
                lock_file, None, f"{reason}, unchanged for {MALFORMED_LOCK_AGE_THRESHOLD:g} seconds"
            )
        if deadline is None:
            deadline = time.monotonic() + _SENTINEL_READ_GRACE
        if state == "exited" or time.monotonic() >= deadline:
            raise SoftFileLockProtocolError(lock_file, None, reason)
        time.sleep(_CLAIM_READ_RETRY)


def _inspect_lock_path(path: Path) -> tuple[int | None, _OccupantState, str]:
    # "held" may be a soft-lock owner that still runs, "exited" is one this host proves gone, "unparsable" is content no
    # lock parses, and "damaged" is a node no lock writes.
    if not stat.S_ISREG(mode := path.lstat().st_mode):
        return None, "damaged", _NODE_DESCRIPTIONS.get(stat.S_IFMT(mode), "a special file")
    try:
        fd, record = _open_record(path, _CLAIM_RECORD_LIMIT)
    except FileNotFoundError:
        raise
    except PermissionError as error:
        # Windows denies the read while a legacy holder keeps its marker open, and a holder running as another user
        # may leave it unreadable here, so a marker that cannot be read may still be held.
        return None, "held", f"a file it cannot read ({error.strerror or error})"
    except OSError as error:
        return None, "damaged", f"a file it cannot read ({error.strerror or error})"
    if record == _LEGACY_SENTINEL:
        return fd, "sentinel", "the strict sentinel"
    os.close(fd)
    return None, *_describe_soft_marker(record.decode("utf-8", errors="replace"))


def _describe_soft_marker(content: str) -> tuple[_OccupantState, str]:
    # A SoftFileLock, MarkerSoftFileLock or SoftFileLease owner holds the path legitimately until it releases, so its
    # marker is contention. One this host proves gone is named, never removed: strict never acts on an inferred death.
    if (record := parse_marker(content)) is not None:
        pid, hostname, start = record.pid, record.hostname, record.start
    elif (holder := _parse_lock_holder(content)) is not None:
        pid, hostname, start = holder
    else:
        return "unparsable", "a file that is neither the sentinel nor a soft-lock marker it can parse"
    if owner_is_stale(pid, hostname, start):
        return "exited", f"a soft-lock marker left by pid {pid} on {hostname}, which has exited"
    return "held", f"a soft-lock marker held by pid {pid} on {hostname}"


def _read_claims(lock_file: str, directory: Path, *, probe_clock: bool) -> tuple[StrictSoftFileClaim, ...]:
    claims: list[StrictSoftFileClaim] = []
    for name in _list_claim_names(lock_file, directory, probe_clock=probe_clock):
        if (name_parts := _parse_claim_name(name)) is None:
            raise SoftFileLockProtocolError(lock_file, name, "unknown claim name or protocol version")
        if (record := _read_claim_record(lock_file, directory, name, probe_clock=probe_clock)) is not None:
            claims.append(_parse_claim(lock_file, name, name_parts, record))
    return tuple(claims)


def _list_claim_names(lock_file: str, directory: Path, *, probe_clock: bool) -> list[str]:
    try:
        with os.scandir(directory) as entries:
            return _public_claim_names(directory, entries, probe_clock=probe_clock)
    except OSError as error:
        reason = f"cannot list claim directory: {error.strerror or type(error).__name__}"
        raise SoftFileLockProtocolError(lock_file, None, reason) from error


def _read_claim_record(lock_file: str, directory: Path, name: str, *, probe_clock: bool) -> bytes | None:
    # A contended scan can list a claim that is not yet cleanly readable, in two ways that both resolve on a brief
    # retry. Windows holds a claim mid-unlink in a delete-pending state that fails an open with EACCES until the unlink
    # completes. On NFS, a peer that unlinks its own claim leaves this client's cached filehandle stale, so the next
    # open returns ESTALE rather than a clean ENOENT until the lookup revalidates against the server. Retrying re-runs
    # the path lookup, which turns the vanished claim into ENOENT (skip) or reads it if it still exists. A genuinely
    # unreadable record (a locked-down file, an EIO fault) still fails closed. The grace starts at the first pending
    # result rather than before the first attempt, so a slow first read cannot use up the window and leave no retry.
    deadline: float | None = None
    while True:
        record, pending = _attempt_claim_read(lock_file, directory, name)
        if pending is None:
            return record
        if deadline is None:
            deadline = time.monotonic() + _CLAIM_READ_GRACE
        elif time.monotonic() >= deadline:
            if pending.errno == ESTALE:
                # A stale handle that outlives revalidation is a claim the server no longer has (RFC 1813
                # NFS3ERR_STALE): skip it like ENOENT. Skipping a peer's vanished claim can only overcount
                # contention, never free a held lock.
                return None
            # Windows can deny opens after a peer removes the claim from the directory.
            if name not in _list_claim_names(lock_file, directory, probe_clock=probe_clock):
                return None
            reason = f"cannot read claim: {pending.strerror or str(pending) or type(pending).__name__}"
            raise SoftFileLockProtocolError(lock_file, name, reason) from pending
        time.sleep(_CLAIM_READ_RETRY)


def _attempt_claim_read(lock_file: str, directory: Path, name: str) -> tuple[bytes | None, OSError | None]:
    # Return the record, or (None, None) when the claim has already gone, or (None, error) for an open the caller may
    # retry: a Windows delete-pending EACCES that resolves to a clean removal, or an NFS ESTALE from a peer unlinking
    # its own claim out from under this client's cached filehandle. Any other OSError is a real fault and fails closed.
    try:
        return _read_record(directory / name, _CLAIM_RECORD_LIMIT), None
    except FileNotFoundError:
        return None, None
    except PermissionError as error:
        return None, error
    except OSError as error:
        if error.errno == ESTALE:
            return None, error
        reason = f"cannot read claim: {error.strerror or str(error) or type(error).__name__}"
        raise SoftFileLockProtocolError(lock_file, name, reason) from error


def _public_claim_names(directory: Path, entries: Iterator[os.DirEntry[str]], *, probe_clock: bool) -> list[str]:
    names: list[str] = []
    now: Callable[[], float | None] = (
        functools.cache(
            functools.partial(
                _filesystem_now, directory, _claim_name("held", secrets.token_hex(_TOKEN_HEX_LENGTH // 2))
            )
        )
        if probe_clock
        else lambda: None
    )
    for entry in entries:
        if not entry.name.startswith("."):
            names.append(entry.name)
        elif (public_name := _private_public_name(entry.name)) is not None and _parse_claim_name(
            public_name
        ) is not None:
            _reclaim_private_record((os.fspath(directory), None), entry.name, now)
    return sorted(names)


def _read_existing_claims(
    lock_file: str, directory: Path, *, probe_clock: bool = True
) -> tuple[StrictSoftFileClaim, ...]:
    if not directory.exists():
        return ()
    return _read_claims(lock_file, directory, probe_clock=probe_clock)


def _parse_claim(
    lock_file: str,
    name: str,
    name_parts: tuple[StrictSoftFileClaimState, str],
    record: bytes,
) -> StrictSoftFileClaim:
    try:
        magic, token, pid_text, hostname_hex, start_text, trailing = record.decode("ascii").split("\n")
        pid = int(pid_text)
        start = int(start_text) if start_text else None
        hostname = bytes.fromhex(hostname_hex).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise SoftFileLockProtocolError(lock_file, name, "malformed claim record") from error
    if not all((
        not trailing,
        magic == _CLAIM_MAGIC,
        token == name_parts[1],
        1 <= pid <= 2**31 - 1,
        str(pid) == pid_text,
        start is None or (start >= 0 and str(start) == start_text),
        hostname.encode().hex() == hostname_hex
        and hostname.isprintable()
        and not any(character.isspace() for character in hostname),
    )):
        raise SoftFileLockProtocolError(lock_file, name, "malformed claim record")
    return StrictSoftFileClaim(name=name, state=name_parts[0], token=token, pid=pid, hostname=hostname, start=start)


def _parse_claim_name(name: str) -> tuple[StrictSoftFileClaimState, str] | None:
    if not name.endswith(".claim"):
        return None
    parts = name.removesuffix(".claim").split("-")
    if len(parts) != _CLAIM_NAME_PART_COUNT or parts[0] not in _CLAIM_STATES or parts[1] != "v1":
        return None
    token = parts[2]
    if len(token) != _TOKEN_HEX_LENGTH or any(character not in "0123456789abcdef" for character in token):
        return None
    return cast("StrictSoftFileClaimState", parts[0]), token


def _claim_name(state: StrictSoftFileClaimState, token: str) -> str:
    return f"{state}-v1-{token}.claim"


def _claim_token_key(name: str) -> str:
    return name.removeprefix("held-").removeprefix("intent-")


def _claim_record(token: str) -> bytes:
    hostname_hex = host_name().encode().hex()
    start = process_start_token(os.getpid())
    start_text = "" if start is None else str(start)
    return f"{_CLAIM_MAGIC}\n{token}\n{os.getpid()}\n{hostname_hex}\n{start_text}\n".encode("ascii")


def _publish_record(
    public_path: str,
    record: bytes,
    mode: int,
) -> BaseException | None:
    directory, public_name = os.path.split(public_path)
    directory = directory or os.curdir
    private_name = _private_record_name(public_name)
    directory_fd = _open_directory(directory) if _OPEN_SUPPORTS_DIR_FD else None
    directory_ref = directory, directory_fd
    try:
        _publish_record_in_directory(directory_ref, (private_name, public_name), mode, record)
    except BaseException as publication_error:  # preserve publication and directory cleanup errors
        try:
            if directory_fd is not None:  # pragma: needs dir-fd
                os.close(directory_fd)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # pragma: needs dir-fd  # preserve publication and directory cleanup errors
            _raise_cleanup_errors("strict publication directory cleanup failed", publication_error, close_error)
        raise
    if directory_fd is not None:  # pragma: needs dir-fd
        try:
            os.close(directory_fd)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # a claim's caller recorded its path before publishing
            return close_error
    return None


def _publish_record_in_directory(
    directory_ref: tuple[str, int | None],
    names: tuple[str, str],
    mode: int,
    record: bytes,
) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_BINARY
    if (o_nofollow := getattr(os, "O_NOFOLLOW", None)) is not None:  # pragma: needs o-nofollow
        flags |= o_nofollow
    private_fd = _open_relative(directory_ref, names[0], flags, mode)
    private_identity: tuple[int, int] | None = None
    try:
        private_identity = _file_identity(os.fstat(private_fd))
        write_all(private_fd, record)
        _link_private_record(directory_ref, names, private_identity)
    except BaseException as publication_error:  # preserve publication and cleanup errors
        close_error, unlink_error = _close_and_unlink_private_record(
            directory_ref,
            names[0],
            private_fd,
            private_identity,
        )
        if close_error is not None or unlink_error is not None:
            _raise_cleanup_errors(
                "strict record publication cleanup failed",
                publication_error,
                close_error,
                unlink_error,
            )
        raise
    close_error, unlink_error = _close_and_unlink_private_record(
        directory_ref,
        names[0],
        private_fd,
        private_identity,
    )
    if close_error is not None or unlink_error is not None:
        _raise_record_finalization_errors(close_error, unlink_error)


def _close_and_unlink_private_record(
    directory_ref: tuple[str, int | None],
    private_name: str,
    private_fd: int,
    private_identity: tuple[int, int] | None,
) -> tuple[BaseException | None, BaseException | None]:
    close_error: BaseException | None = None
    try:
        os.close(private_fd)
    except BaseException as error:  # ruff:ignore[blind-except]  # returned for grouping with unlink failures
        close_error = error
    unlink_error: BaseException | None = None
    try:
        if private_identity is None:
            _unlink_relative(directory_ref, private_name)
        else:
            _unlink_relative_if_identity(directory_ref, private_name, private_identity)
    except FileNotFoundError:
        pass
    except BaseException as error:  # ruff:ignore[blind-except]  # returned for grouping with close failures
        unlink_error = error
    return close_error, unlink_error


def _link_private_record(
    directory_ref: tuple[str, int | None],
    names: tuple[str, str],
    private_identity: tuple[int, int],
) -> None:
    try:
        _link_or_find_committed(directory_ref, *names, private_identity)
    except FileNotFoundError as error:
        if _relative_identity(directory_ref, names[0]) is not None:
            raise
        msg = "private publication record was reclaimed"
        raise _PrivateRecordReclaimedError(msg) from error
    if _relative_identity(directory_ref, names[1]) == private_identity:
        return
    msg_0 = "private publication record was replaced"
    raise _PrivateRecordReclaimedError(msg_0)


def _raise_record_finalization_errors(
    close_error: BaseException | None,
    unlink_error: BaseException | None,
) -> None:
    errors = [error for error in (close_error, unlink_error) if error is not None]
    if len(errors) > 1:
        _raise_cleanup_errors("strict record finalization failed", errors[0], *errors[1:])
    raise errors[0]


def _private_record_name(public_name: str) -> str:
    return f".{public_name}{_PRIVATE_RECORD_MARKER}{secrets.token_hex(_PRIVATE_RECORD_RANDOM_HEX_LENGTH // 2)}.tmp"


def _private_public_name(private_name: str) -> str | None:
    if not private_name.startswith(".") or not private_name.endswith(_PRIVATE_RECORD_SUFFIX):
        return None
    public_name, marker, random_hex = private_name[1 : -len(_PRIVATE_RECORD_SUFFIX)].rpartition(_PRIVATE_RECORD_MARKER)
    if (
        marker != _PRIVATE_RECORD_MARKER
        or len(random_hex) != _PRIVATE_RECORD_RANDOM_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in random_hex)
    ):
        return None
    return public_name


def _reclaim_sentinel_private_records(path: Path) -> None:
    directory_ref = os.fspath(path.parent), None
    now = functools.cache(functools.partial(_filesystem_now, path.parent, path.name))
    with os.scandir(path.parent) as entries:
        for entry in entries:
            if _private_public_name(entry.name) == path.name:
                _reclaim_private_record(directory_ref, entry.name, now)


def _reclaim_private_record(
    directory_ref: tuple[str, int | None], private_name: str, now: Callable[[], float | None]
) -> None:
    directory, directory_fd = directory_ref
    try:
        private_stat = (
            os.stat(private_name, dir_fd=directory_fd, follow_symlinks=False)
            if directory_fd is not None and _STAT_SUPPORTS_DIR_FD
            else Path(directory, private_name).lstat()
        )
    except FileNotFoundError:
        return
    if not stat.S_ISREG(private_stat.st_mode):
        msg = f"{Path(directory, private_name)} is not a regular private record"
        raise OSError(msg)
    if private_stat.st_nlink == 1 and (
        (current := now()) is None or current - private_stat.st_mtime < _PRIVATE_RECORD_GRACE
    ):
        return
    try:
        _unlink_private_record_once(directory_ref, private_name)
    except FileNotFoundError:
        pass
    except OSError as error:
        if error.errno not in {EACCES, EPERM}:
            raise


def _filesystem_now(directory: Path, public_name: str) -> float | None:
    # A network filesystem stamps mtimes with the server's clock, which can run seconds off this client's, so read "now"
    # as the mtime of a probe written like a record (liblockfile uses a read's atime). The probe has a private-record
    # name, so a scan reaps one left behind. Without a stamp no age is trusted.
    probe = directory / _private_record_name(public_name)
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        return None
    try:
        write_all(fd, b"\n")
        return os.fstat(fd).st_mtime
    except OSError:
        return None
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            probe.unlink()


def _unlink_private_record_once(directory_ref: tuple[str, int | None], private_name: str) -> None:
    directory, directory_fd = directory_ref
    if (
        directory_fd is not None and _UNLINK_SUPPORTS_DIR_FD
    ):  # pragma: no cover  # callers always pass directory_fd=None
        os.unlink(private_name, dir_fd=directory_fd)
    else:
        Path(directory, private_name).unlink()


def _relative_identity(directory_ref: tuple[str, int | None], name: str) -> tuple[int, int] | None:
    directory, directory_fd = directory_ref
    try:
        if directory_fd is not None and _STAT_SUPPORTS_DIR_FD:  # pragma: needs dir-fd
            path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        else:  # pragma: win32 cover
            path_stat = Path(directory, name).lstat()
    except FileNotFoundError:
        return None
    return _file_identity(path_stat)


def _unlink_relative_if_identity(
    directory_ref: tuple[str, int | None],
    name: str,
    identity: tuple[int, int],
) -> None:
    if _relative_identity(directory_ref, name) == identity:
        with contextlib.suppress(FileNotFoundError):
            _unlink_relative(directory_ref, name)


def _open_relative(directory_ref: tuple[str, int | None], name: str, flags: int, mode: int) -> int:
    directory, directory_fd = directory_ref
    return (
        os.open(name, flags, mode, dir_fd=directory_fd)
        if directory_fd is not None
        else os.open(Path(directory, name), flags, mode)
    )


def _link_relative(directory_ref: tuple[str, int | None], source_name: str, destination_name: str) -> None:
    directory, directory_fd = directory_ref
    if directory_fd is not None and _LINK_SUPPORTS_DIR_FD:  # pragma: needs dir-fd
        _link_no_follow(source_name, destination_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        return
    _link_no_follow(Path(directory, source_name), Path(directory, destination_name))  # pragma: win32 cover


def _link_or_find_committed(
    directory_ref: tuple[str, int | None],
    source_name: str,
    destination_name: str,
    source_identity: tuple[int, int] | None = None,
) -> None:
    try:
        _link_relative(directory_ref, source_name, destination_name)
    except FileExistsError:
        # NFS retransmits a LINK whose reply was lost and reports EEXIST for the committed link; as in open(2) NOTES
        # and liblockfile, it succeeded when the destination names the source. Prefer the fstat-ed identity: a peer's
        # scan reaps a private record once it has two links, so its name may already resolve to nothing.
        if (identity := _relative_identity(directory_ref, destination_name)) is None or identity != (
            source_identity or _relative_identity(directory_ref, source_name)
        ):
            raise


def _link_no_follow(
    source: str | Path,
    destination: str | Path,
    *,
    src_dir_fd: int | None = None,
    dst_dir_fd: int | None = None,
) -> None:
    if not _HAS_LINK:
        # No os.link at all (Termux/Android): report it as unsupported like a filesystem that refuses hard links, so
        # the acquire path raises SoftFileLockProtocolError instead of a bare AttributeError.
        msg = "os.link is unavailable on this platform"
        raise NotImplementedError(msg)
    # The source is a private record this process created with O_CREAT | O_EXCL, so follow_symlinks guards nothing an
    # attacker can reach. Pass the option only when the runtime honors it: PyPy advertises it through
    # os.supports_follow_symlinks yet its linkat rejects it with EINVAL, so probe once rather than trust the set.
    if _LINK_HONORS_FOLLOW_SYMLINKS:
        os.link(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=False)
    else:  # pragma: lacks link-follow-symlinks
        os.link(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)


def _unlink_relative(directory_ref: tuple[str, int | None], name: str) -> None:
    directory, directory_fd = directory_ref
    if directory_fd is not None and _UNLINK_SUPPORTS_DIR_FD:  # pragma: needs dir-fd
        os.unlink(name, dir_fd=directory_fd)
    elif sys.platform == "win32":  # pragma: win32 cover
        _unlink_in_directory(Path(directory), name)
    else:
        Path(directory, name).unlink()  # pragma: win32 no cover


def _link_no_replace(directory: Path, source_name: str, destination_name: str) -> BaseException | None:
    if _LINK_SUPPORTS_DIR_FD:  # pragma: needs dir-fd
        directory_fd = _open_directory(str(directory))
        try:
            _link_or_find_committed((str(directory), directory_fd), source_name, destination_name)
        except BaseException as link_error:  # preserve link and directory cleanup errors
            try:
                os.close(directory_fd)
            except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve link and directory cleanup errors
                _raise_cleanup_errors("strict link directory cleanup failed", link_error, close_error)
            raise
        try:
            os.close(directory_fd)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # the caller recorded the held path before linking
            return close_error
        return None
    _link_or_find_committed((str(directory), None), source_name, destination_name)  # pragma: win32 cover
    return None  # pragma: win32 cover


def _unlink_owner_path(path: str) -> BaseException | None:
    directory, name = os.path.split(path)
    return _unlink_in_directory(Path(directory or os.curdir), name)


def _unlink_owner_paths(paths: tuple[str, ...]) -> tuple[list[str], list[BaseException]]:
    results = [(path, _unlink_owner_path_result(path)) for path in paths]
    return [path for path, (removed, _) in results if not removed], [
        error for _, (_, error) in results if error is not None
    ]


def _unlink_owner_path_result(path: str) -> tuple[bool, BaseException | None]:
    try:
        cleanup_error = _unlink_owner_path(path)
    except FileNotFoundError:
        return True, None
    except BaseException as error:  # ruff:ignore[blind-except]  # keep ownership when unlink did not commit
        return False, error
    return True, cleanup_error


def _raise_recorded_errors(message: str, errors: list[BaseException]) -> None:
    if len(errors) > 1:
        _raise_cleanup_errors(message, errors[0], *errors[1:])
    raise errors[0]


def _unlink_in_directory(directory: Path, name: str) -> BaseException | None:
    if _UNLINK_SUPPORTS_DIR_FD:  # pragma: needs dir-fd
        directory_fd = _open_directory(str(directory))
        try:
            os.unlink(name, dir_fd=directory_fd)
        except BaseException as unlink_error:  # preserve unlink and directory cleanup errors
            try:
                os.close(directory_fd)
            except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve unlink and directory cleanup errors
                _raise_cleanup_errors("strict unlink directory cleanup failed", unlink_error, close_error)
            raise
        try:
            os.close(directory_fd)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # caller commits the removed path before raising
            return close_error
        return None
    if sys.platform != "win32":  # pragma: win32 no cover
        Path(directory / name).unlink()
        return None
    retry_delay = 0.001  # pragma: win32 cover
    for attempt in range(_UNLINK_MAX_RETRIES):  # pragma: win32 cover
        if (error := _unlink_error(directory / name)) is None:
            return None
        if error.errno not in {EACCES, EPERM} or attempt == _UNLINK_MAX_RETRIES - 1:
            raise error
        time.sleep(retry_delay)
        retry_delay *= 2
    return None  # pragma: no cover  # the final retry returns or raises


def _unlink_error(path: Path) -> OSError | None:  # pragma: win32 cover
    try:
        path.unlink()
    except OSError as error:
        return error
    return None


def _open_directory(directory: str) -> int:  # pragma: needs dir-fd
    return os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _read_record(path: Path, limit: int) -> bytes:
    fd, record = _open_record(path, limit)
    os.close(fd)
    return record


def _open_record(path: Path, limit: int) -> tuple[int, bytes]:
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode):
        msg = f"{path} is not a regular file"
        raise OSError(msg)
    flags = os.O_RDONLY | _O_BINARY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        record = _read_opened_record(fd, path, path_stat, limit)
    except BaseException as read_error:  # preserve read and descriptor cleanup errors
        try:
            os.close(fd)
        except BaseException as close_error:  # ruff:ignore[blind-except]  # preserve read and descriptor cleanup errors
            _raise_cleanup_errors("strict record read cleanup failed", read_error, close_error)
        raise
    return fd, record


def _read_opened_record(fd: int, path: Path, path_stat: os.stat_result, limit: int) -> bytes:
    opened_stat = os.fstat(fd)
    if not stat.S_ISREG(opened_stat.st_mode):
        msg = f"{path} is not a regular file"
        raise OSError(msg)
    if _file_identity(opened_stat) != _file_identity(path_stat):
        msg = f"{path} changed while opening"
        raise OSError(msg)
    record = os.read(fd, limit + 1)
    if len(record) > limit:
        msg = f"{path} exceeds {limit} bytes"
        raise OSError(msg)
    return record


def _ensure_protocol_directory(lock_file: str, directory: Path) -> None:
    try:
        directory.mkdir()
    except FileExistsError:
        try:
            mode = directory.lstat().st_mode
        except OSError as error:
            raise SoftFileLockProtocolError(lock_file, None, f"cannot inspect {directory}") from error
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            return
        raise SoftFileLockProtocolError(lock_file, None, f"{directory} is not a real directory") from None


def _validate_force_break_name(name: str) -> None:
    invalid_component = not name or name.startswith(".") or name in {os.curdir, os.pardir}
    if invalid_component or any(separator in name for separator in ("/", "\\", "\x00")):
        msg = "claim_name must be one public claim basename"
        raise ValueError(msg)


def _require_exact_name(directory: Path, name: str) -> None:
    with os.scandir(directory) as entries:
        if not any(entry.name == name for entry in entries):
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), name)


def _raise_if_hard_links_unsupported(lock_file: str, error: NotImplementedError | OSError) -> None:
    unsupported = (
        isinstance(error, NotImplementedError)
        or error.errno in _HARD_LINK_UNSUPPORTED_ERRNOS
        or (getattr(error, "winerror", None) in _WINDOWS_HARD_LINK_UNSUPPORTED)
    )
    if unsupported:
        reason = "filesystem does not support atomic no-replace hard-link publication"
        raise SoftFileLockProtocolError(lock_file, None, reason) from error


def _file_identity(st: os.stat_result) -> tuple[int, int]:
    return st.st_dev, st.st_ino


__all__ = [
    "StrictSoftFileClaim",
    "StrictSoftFileClaimState",
    "StrictSoftFileLock",
]
<|endoftext|>"""Separate caller cancellation from backend task and executor-future results."""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Final, Generic, Literal, NoReturn, TypeVar, cast
from weakref import WeakKeyDictionary

from ._api import _append_exception_context, _fork_transition, _raise_chained_errors, _resolve_read_write_timeout
from ._error import Timeout

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from concurrent.futures import Executor

    from ._read_write import ReadWriteLock
    from ._soft_rw import SoftReadWriteLock

_T = TypeVar("_T")


class _AsyncTransitionUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class _BackendOutcome(Generic[_T]):
    value: _T | None = None
    error: BaseException | None = None


class _AsyncTransitionGate:
    def __init__(self) -> None:
        self._tail_lock: Final[Lock] = Lock()
        self._tail: ConcurrentFuture[None] | None = None

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncGenerator[None]:
        ticket: ConcurrentFuture[None] = ConcurrentFuture()
        with self._tail_lock:
            predecessor = self._tail
            self._tail = ticket
        if predecessor is not None:
            try:
                await _wait_until_done(asyncio.wrap_future(predecessor))
            except BaseException:
                predecessor.add_done_callback(lambda _predecessor: self._leave(ticket))
                raise
        try:
            yield
        finally:
            self._leave(ticket)

    @contextlib.asynccontextmanager
    async def hold_for_acquire(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        deadline: float | None,
        poll_interval: float,
    ) -> AsyncGenerator[None]:
        ticket: ConcurrentFuture[None] = ConcurrentFuture()
        with self._tail_lock:
            predecessor = self._tail
            self._tail = ticket
        if predecessor is not None and not predecessor.done():
            try:
                await self._wait_for_predecessor(
                    predecessor,
                    blocking=blocking,
                    cancel_check=cancel_check,
                    deadline=deadline,
                    poll_interval=poll_interval,
                )
            except BaseException:
                predecessor.add_done_callback(lambda _predecessor: self._leave(ticket))
                raise
        try:
            yield
        finally:
            self._leave(ticket)

    @staticmethod
    async def _wait_for_predecessor(
        predecessor: ConcurrentFuture[None],
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        deadline: float | None,
        poll_interval: float,
    ) -> None:
        if not blocking:
            raise _AsyncTransitionUnavailableError
        waiter = asyncio.wrap_future(predecessor)
        while not predecessor.done():
            if cancel_check is not None and cancel_check():
                raise _AsyncTransitionUnavailableError
            if deadline is not None:
                if (remaining := deadline - time.perf_counter()) <= 0:
                    raise _AsyncTransitionUnavailableError
                wait_interval = min(poll_interval, remaining) if cancel_check is not None else remaining
            else:
                wait_interval = poll_interval if cancel_check is not None else None
            await asyncio.wait((waiter,), timeout=wait_interval)

    def _leave(self, ticket: ConcurrentFuture[None]) -> None:
        with self._tail_lock:
            if self._tail is ticket:
                self._tail = None
        ticket.set_result(None)


class _TaskOwners:
    """
    Per-task ownership in front of a sync reader/writer lock that tracks threads.

    An async wrapper runs each call on an executor thread, and that thread does not identify the calling task. Waiting
    writers block new readers so a stream of readers cannot starve them. A thread lock guards the table because tasks on
    different event loops share it.
    """

    def __init__(self) -> None:
        self._lock: Final[Lock] = Lock()
        self._mode: Literal["read", "write"] | None = None
        self._depths: dict[asyncio.Task[object], int] = {}
        self._transitioning = False
        self._waiting_writers = 0
        self._changed: ConcurrentFuture[None] = ConcurrentFuture()
        #: The executor thread that took the sync lock's level for every task, which may leave on another worker.
        self.backend_thread: int | None = None

    async def acquire(
        self,
        mode: Literal["read", "write"],
        *,
        timeout: float,
        blocking: bool,
        lock_file: str,
        enter: Callable[[float], Awaitable[None]],
    ) -> None:
        task = _current_task()
        timeout = _resolve_read_write_timeout(timeout, blocking=blocking)
        deadline = None if not blocking or timeout < 0 else time.perf_counter() + timeout
        waiting_writer = False
        try:
            while isinstance(
                admission := self._admit(task, mode, lock_file, waiting_writer=waiting_writer), ConcurrentFuture
            ):
                if not blocking or (deadline is not None and time.perf_counter() >= deadline):
                    raise Timeout(lock_file)
                if mode == "write" and not waiting_writer:
                    waiting_writer = True
                    with self._lock:
                        self._waiting_writers += 1
                wait = None if deadline is None else deadline - time.perf_counter()
                await asyncio.wait((asyncio.wrap_future(admission),), timeout=wait)
        finally:
            if waiting_writer:
                with self._lock:
                    self._waiting_writers -= 1
                    self._notify()
        if admission == "held":
            return
        try:
            await enter(-1 if deadline is None else max(0.0, deadline - time.perf_counter()))
        except BaseException:
            self._finish_transition(mode=None)
            raise
        self._finish_transition(mode=mode, holder=task)

    def _admit(
        self,
