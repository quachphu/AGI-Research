    # A per-key lock collapses concurrent workers into a single token request per expired/missing
    # key, without blocking fetches for other keys (e.g. other repos downloaded in parallel).
    with _key_lock(cache_key):
        cached_info = XET_CONNECTION_INFO_CACHE.get(cache_key)
        if cached_info is not None and not _is_expired(cached_info):
            return cached_info

        # Fetch from server
        resp = http_backoff("GET", url, headers=headers)
        hf_raise_for_status(resp)

        metadata = parse_xet_connection_info_from_headers(resp.headers)
        if metadata is None:
            raise ValueError("Xet headers have not been correctly set by the server.")

        with _XET_CONNECTION_INFO_LOCK:
            # Delete expired cache entries
            for k, v in list(XET_CONNECTION_INFO_CACHE.items()):
                if _is_expired(v):
                    XET_CONNECTION_INFO_CACHE.pop(k, None)

            # Enforce cache size limit
            if len(XET_CONNECTION_INFO_CACHE) >= XET_CONNECTION_INFO_CACHE_SIZE:
                XET_CONNECTION_INFO_CACHE.pop(next(iter(XET_CONNECTION_INFO_CACHE)))

            # Update cache
            XET_CONNECTION_INFO_CACHE[cache_key] = metadata

        return metadata


def _cache_key(url: str, headers: dict[str, str]) -> str:
    """Return a unique cache key for the given request parameters."""
    lower_headers = {k.lower(): v for k, v in headers.items()}  # casing is not guaranteed here
    auth_header = lower_headers.get("authorization", "")
    return f"{url}|{auth_header}"


def _key_lock(cache_key: str) -> threading.Lock:
    """Return the lock guarding token fetches for `cache_key`, creating it if needed."""
    with _XET_CONNECTION_INFO_LOCK:
        return _XET_CONNECTION_INFO_KEY_LOCKS.setdefault(cache_key, threading.Lock())


def _is_expired(connection_info: XetConnectionInfo) -> bool:
    """Check if the given XET connection info is expired."""
    return connection_info.expiration_unix_epoch <= int(time.time()) + XET_CONNECTION_INFO_SAFETY_PERIOD


@validate_hf_hub_args
def xet_connection_info_refresh_url(
    *,
    token_type: XetTokenType,
    repo_id: str,
    repo_type: str,
    revision: str | None = None,
    endpoint: str | None = None,
) -> str:
    """
    Build the URL used to fetch or refresh a Xet access token for a given repo.
    Args:
        token_type (`XetTokenType`):
            Type of the token to request: `"read"` or `"write"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        repo_type (`str`):
            Type of the repo (e.g. `"model"`, `"dataset"`, `"space"`, `"bucket"`).
        revision (`str`, `optional`):
            The revision of the repo to get the token for.
        endpoint (`str`, `optional`):
            The endpoint to use for the request. Defaults to the Hub endpoint.
    Returns:
        `str`:
            The fully-qualified URL of the token refresh endpoint.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    url = f"{endpoint}/api/{repo_type}s/{repo_id}/xet-{token_type.value}-token"
    if repo_type != "bucket" or revision is not None:
        # On "bucket" repo type, the revision never needed => don't use it
        # Otherwise, use the revision.
        # Note: when creating a PR on a git-based repo, user needs write access but they don't know the revision in advance.
        # => pass "/None" in URL and server will return a token for PR refs.
        url += f"/{revision}"
    return url


class XetSessionHolder:
    """Holds an optional XetSession; supports safe re-creation after sigint_abort or fork.

    Thread-safe: a ``threading.Lock`` guards all state mutations, which matters
    for free-threaded Python (3.14t) where multiple threads can race on ``get()``
    or ``sigint_abort()`` without the GIL serialising them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Any = None
        self._session_pid: int | None = None

    def get(self):
        """Return the current session, creating one if needed.

        Fork-safe: if the current process PID differs from the PID that created
        the session (i.e. we are in a forked child), the old session is discarded
        and a fresh session is created for this process.
        """
        with self._lock:
            current_pid = os.getpid()

            if self._session is not None and self._session_pid != current_pid:
                # Fork detected. Discard the parent's session; the Rust Drop will
                # call discard_runtime() (std::mem::forget) rather than the normal
                # shutdown path, so this returns immediately without blocking.
                self._session = None

            if self._session is None:
                from hf_xet import XetSession

                self._session = XetSession()
                self._session_pid = current_pid

            return self._session

    def sigint_abort(self):
        """Abort the current session and clear it so the next get() creates a fresh one."""
        with self._lock:
            if self._session is not None:
                try:
                    self._session.sigint_abort()
                except Exception:
                    pass
                self._session = None
                self._session_pid = None


_GLOBAL_XET_HOLDER = XetSessionHolder()


def get_xet_session():
    """Return the global :class:`hf_xet.XetSession`, creating it on first call.

    The session is shared across all calls within a process, just as the HTTP
    client returned by :func:`~huggingface_hub.utils._http.get_session` is shared.
    It is created lazily and is fork-safe and thread-safe.
    """
    return _GLOBAL_XET_HOLDER.get()


def xet_headers_without_auth(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with the authorization header removed.

    Xet storage requests use a short-lived xet access token for auth, so the
    Hub authorization header must not be forwarded to xet storage endpoints.
    """
    return {key: value for key, value in headers.items() if key.lower() != "authorization"}


def abort_xet_session():
    """Abort the global xet session after a KeyboardInterrupt.

    Cancels any in-flight Rust operation and clears the session so the next
    call to :func:`get_xet_session` starts fresh (notebook-friendly).
    """
    _GLOBAL_XET_HOLDER.sigint_abort()
<|endoftext|>from collections import OrderedDict
from typing import Any

from . import is_google_colab, is_notebook
from .tqdm import _create_progress_bar, tqdm


def _format_speed_postfix(speed: float | None) -> str:
    s = tqdm.format_sizeof(speed) if speed is not None else "???"
    return f"{s}B/s  ".rjust(10, " ")


# Transfer byte count is hard to predict (dedup/compression), so we omit a total and show bytes only.
XET_TRANSFER_BAR_FORMAT = "{desc}: {bar}| {n_fmt:>5}B{postfix:>12}"
XET_BYTES_BAR_FORMAT = "{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B{postfix:>12}"

_NUM_OVERVIEW_BARS = 2  # data_processing + upload (always shown)
_NUM_SHARD_BARS = 1  # single Validating bar (lazy; covers shard upload/validate/sync)


def _set_monotonic_total(bar, total: int | None) -> None:
    if total is None or not hasattr(bar, "total"):
        return
    bar.total = max(bar.total or 0, total)


def _update_transfer_bar(bar, inc: int) -> None:
    """Update the transfer bar and grow its hidden total so the bar graphic advances.

    Network bytes are hard to predict (dedup/compression), so the display omits a denominator.
    tqdm still needs an internal total for the bar width — seeded from file size when known,
    then expanded here if bytes received exceed that estimate.
    """
    n_after = getattr(bar, "n", 0) + inc
    current_total = getattr(bar, "total", 0) or 0
    if n_after > 0 and current_total < n_after:
        bar.total = max(current_total, int(n_after * 1.25) + 1)
    bar.update(inc)


def _finish_transfer_bar(bar) -> None:
    """Snap the transfer bar to 100% when downloading stops.

    Transfer totals are seeded from file size, but actual network bytes are often lower.
    Set ``total = n`` so the bar fills completely instead of stopping partway.
    """
    n = getattr(bar, "n", 0)
    if n > 0 and hasattr(bar, "total") and bar.total != n:
        bar.total = n
        bar.refresh()


def _set_aggregate_rate_postfix(bar) -> None:
    """Show a shared bar's own throughput as its rate.

    When many files feed one bar (snapshot download), each file only knows its own per-item rate — a
    fraction of the total. Deriving the rate from the shared bar's aggregated byte count reports the
    true combined speed instead of whichever single file reported last.
    """
    bar.set_postfix_str(_format_speed_postfix(bar.format_dict.get("rate")), refresh=False)


class XetDownloadProgressReporter:
    """Dual progress bars for Xet downloads: network transfer and file reconstruction.

    ``total_transfer_bytes_completed`` tracks bytes received from the network (updated continuously).
    ``total_bytes_completed`` tracks bytes written to disk (updated after buffered chunks are flushed).
    Showing both bars gives responsive feedback on slow connections where reconstruction lags behind transfer.
    """

    def __init__(
        self,
        *,
        reconstruction_desc: str,
        transfer_desc: str = "Downloading bytes",
        total: int | None = None,
        log_level: int,
        name: str | None = None,
        tqdm_class: type | None = None,
        external_reconstruction_bar: Any | None = None,
        position: int = 0,
    ):
        self._prev_bytes_completed = 0
        self._prev_transfer_bytes_completed = 0

        cls = tqdm_class or tqdm
        routes_transfer_via_reconstruction = external_reconstruction_bar is not None and callable(
            getattr(external_reconstruction_bar, "update_transfer", None)
        )
        uses_aggregated_tqdm_class = external_reconstruction_bar is None and callable(
            getattr(cls, "update_transfer", None)
        )

        if external_reconstruction_bar is not None:
            self.reconstruction_bar = external_reconstruction_bar
            self._owns_reconstruction_bar = False
        else:
            self.reconstruction_bar = _create_progress_bar(
                cls=cls,  # ty: ignore[invalid-argument-type]
                log_level=log_level,
                name=name,
                desc=reconstruction_desc,
                total=total,
                unit="B",
                unit_scale=True,
                position=position + 1,
                bar_format=XET_BYTES_BAR_FORMAT,
                leave=True,
            )
            self._owns_reconstruction_bar = True

        if routes_transfer_via_reconstruction or uses_aggregated_tqdm_class:
            self.transfer_bar = self.reconstruction_bar
            self._owns_transfer_bar = False
        elif external_reconstruction_bar is not None:
            self.transfer_bar = None
            self._owns_transfer_bar = False
        else:
            self.transfer_bar = _create_progress_bar(
                cls=cls,  # ty: ignore[invalid-argument-type]
                log_level=log_level,
                name=f"{name}.transfer" if name else None,
                desc=transfer_desc,
                total=total,
                unit="B",
                unit_scale=True,
                position=position,
                bar_format=XET_TRANSFER_BAR_FORMAT,
                leave=True,
            )
            self._owns_transfer_bar = True

    @property
    def _aggregated(self) -> bool:
        return self.transfer_bar is not None and self.transfer_bar is self.reconstruction_bar

    def update_progress(self, group_report, _item_reports: dict | None = None) -> None:
        bytes_inc = max(0, group_report.total_bytes_completed - self._prev_bytes_completed)
        transfer_inc = max(0, group_report.total_transfer_bytes_completed - self._prev_transfer_bytes_completed)
        self._prev_bytes_completed = group_report.total_bytes_completed
        self._prev_transfer_bytes_completed = group_report.total_transfer_bytes_completed

        if bytes_inc > 0:
            self.reconstruction_bar.update(bytes_inc)
            self.reconstruction_bar.set_postfix_str(
                _format_speed_postfix(group_report.total_bytes_completion_rate), refresh=False
            )

        if transfer_inc > 0 and self.transfer_bar is not None:
            if self._aggregated:
                self.reconstruction_bar.update_transfer(transfer_inc)
                self.reconstruction_bar.set_transfer_postfix_str(
                    _format_speed_postfix(group_report.total_transfer_bytes_completion_rate), refresh=False
                )
            else:
                _update_transfer_bar(self.transfer_bar, transfer_inc)
                self.transfer_bar.set_postfix_str(
                    _format_speed_postfix(group_report.total_transfer_bytes_completion_rate), refresh=False
                )

        if group_report.total_bytes:
            _set_monotonic_total(self.reconstruction_bar, group_report.total_bytes)

    def close(self) -> None:
        """Close bars owned by this reporter.

        Standalone downloads finish the transfer bar first (snap hidden total to ``n``), then close it.
        Aggregated and external bars (e.g. snapshot ``_AggregatedTqdm`` or a reused ``_tqdm_bar``) are left
        open — their parent caller owns their lifecycle.
        """
        if self.transfer_bar is not None and self._owns_transfer_bar:
            _finish_transfer_bar(self.transfer_bar)
            if hasattr(self.transfer_bar, "close"):
                self.transfer_bar.close()
        if self._owns_reconstruction_bar and hasattr(self.reconstruction_bar, "close"):
            self.reconstruction_bar.close()

    def __enter__(self) -> "XetDownloadProgressReporter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class XetUploadProgressReporter:
    """
    Reports on progress for Xet uploads.

    Shows summary progress bars when running in notebooks or GUIs, and detailed per-file progress in console environments.
    """

    def __init__(self, n_lines: int = 10, description_width: int = 30, total_files: int | None = None):
        self.n_lines = n_lines
        self.description_width = description_width
        self.total_files = total_files

        self.per_file_progress = is_google_colab() or not is_notebook()

        _nrows = (
            _NUM_OVERVIEW_BARS + n_lines + _NUM_SHARD_BARS + 1
            if self.per_file_progress
            else _NUM_OVERVIEW_BARS + _NUM_SHARD_BARS + 1
        )

        self.tqdm_settings = {
            "unit": "B",
            "unit_scale": True,
            "leave": True,
            "unit_divisor": 1000,
            "nrows": _nrows,
            "miniters": 1,
            "bar_format": XET_BYTES_BAR_FORMAT,
        }

        # Percentage only — validation-entry totals are opaque Xet internals.
        self.percent_tqdm_settings = {
            "unit": "",
            "unit_scale": False,
            "leave": True,
            "nrows": _nrows,
            "miniters": 1,
            "bar_format": "{desc}: {bar}| {percentage:3.0f}%{postfix:>12}",
        }

        # Overview bars (always shown)
        self.data_processing_bar = tqdm(
            total=0, desc=self.format_desc("Processing Files (0 / 0)", False), position=0, **self.tqdm_settings
        )

        self.upload_bar = tqdm(
            total=0, desc=self.format_desc("New Data Upload", False), position=1, **self.tqdm_settings
        )

        # Single Validating bar created lazily on the first shard event.
        # Covers shard upload / validation / sync without exposing "shard" jargon.
        # Positioned dynamically after the active per-file bars (see _validating_bar_position).
        self.validating_bar: tqdm | None = None

        self.known_items: set[str] = set()
        self.completed_items: set[str] = set()

        # Track previous absolute values to compute increments
        self._prev_bytes_completed: int = 0
        self._prev_transfer_bytes_completed: int = 0

        # Item bars (scrolling view); positions start at _NUM_OVERVIEW_BARS
        self.item_state: OrderedDict[str, Any] = OrderedDict()
        self.current_bars: list = [None] * self.n_lines

    def format_desc(self, name: str, indent: bool) -> str:
        """
        if name is longer than width characters, prints ... at the start and then the last width-3 characters of the name, otherwise
        the whole name right justified into description_width characters.  Also adds some padding.
        """

        if not self.per_file_progress:
            # Here we just use the defaults.
            return name

        padding = "  " if indent else ""
        width = self.description_width - len(padding)

        if len(name) > width:
            name = f"...{name[-(width - 3) :]}"

        return f"{padding}{name.ljust(width)}"

    def _validating_bar_position(self) -> int:
        """Place Validating right after overview bars and any active per-file bars."""
        if not self.per_file_progress:
            return _NUM_OVERVIEW_BARS
        n_active = sum(1 for bar in self.current_bars if bar is not None)
        return _NUM_OVERVIEW_BARS + n_active

    def _init_validating_bar(self) -> None:
        """Create the Validating progress bar on first shard event."""
        self.validating_bar = tqdm(
            total=0,
            desc=self.format_desc("Validating", False),
            position=self._validating_bar_position(),
            **self.percent_tqdm_settings,
        )

    def reset_for_next_commit(self):
        """Reset per-commit state so the reporter can be reused across multiple upload commits."""
        self._prev_bytes_completed = 0
        self._prev_transfer_bytes_completed = 0
        self.known_items.clear()
        self.completed_items.clear()
        self.item_state.clear()
        # Shard counters are absolute per commit — restart Validating at 0% for the next chunk.
        if self.validating_bar is not None:
            self.validating_bar.n = 0
            self.validating_bar.total = 0
            self.validating_bar.refresh()

    def update_progress(self, group_report, item_reports: dict):
        # Update all the per-item values.
        for item in item_reports.values():
            item_name = item.item_name

            self.known_items.add(item_name)

            # Only care about items where the processing has already started.
            if item.bytes_completed == 0:
                continue

            # Overwrite the existing value in there.
            self.item_state[item_name] = item

        bar_idx = 0
        new_completed = []

        # Now, go through and update all the bars
        for name, item in self.item_state.items():
            # Is this ready to be removed on the next update?
            if item.bytes_completed == item.total_bytes:
                self.completed_items.add(name)
                new_completed.append(name)

            # If we're only showing summary information, then don't update the individual bars
            if not self.per_file_progress:
                continue

            # If we've run out of bars to use, then collapse the last ones together.
            if bar_idx >= len(self.current_bars):
                bar = self.current_bars[-1]
                in_final_bar_mode = True
                final_bar_aggregation_count = bar_idx + 1 - len(self.current_bars)
            else:
                bar = self.current_bars[bar_idx]
                in_final_bar_mode = False

            if bar is None:
                # Move Validating before creating a file bar in its current row.
                if self.validating_bar is not None and self.validating_bar.pos == -(_NUM_OVERVIEW_BARS + bar_idx):
                    self.validating_bar.clear()
                    self.validating_bar.pos -= 1
                    self.validating_bar.refresh()
                self.current_bars[bar_idx] = tqdm(
                    desc=self.format_desc(name, True),
                    position=_NUM_OVERVIEW_BARS + bar_idx,
                    total=item.total_bytes,
                    initial=item.bytes_completed,
                    **self.tqdm_settings,
                )

            elif in_final_bar_mode:
                bar.n += item.bytes_completed
                bar.total += item.total_bytes
                bar.set_description(self.format_desc(f"[+ {final_bar_aggregation_count} files]", True), refresh=False)
            else:
                bar.set_description(self.format_desc(name, True), refresh=False)
                bar.n = item.bytes_completed
                bar.total = item.total_bytes

            bar_idx += 1

        # Remove all the completed ones from the ordered dictionary
        for name in new_completed:
            # Only remove ones from consideration to make room for more items coming in.
            if len(self.item_state) <= self.n_lines:
                break

            del self.item_state[name]

        if self.per_file_progress:
            # Now manually refresh each of the bars
            for bar in self.current_bars:
                if bar:
                    bar.refresh()

        # Update overall bars
        bytes_inc = max(0, group_report.total_bytes_completed - self._prev_bytes_completed)
        transfer_inc = max(0, group_report.total_transfer_bytes_completed - self._prev_transfer_bytes_completed)
        self._prev_bytes_completed = group_report.total_bytes_completed
        self._prev_transfer_bytes_completed = group_report.total_transfer_bytes_completed

        self.data_processing_bar.total = group_report.total_bytes
        total_files_count = self.total_files if self.total_files is not None else len(self.known_items)
        self.data_processing_bar.set_description(
            self.format_desc(f"Processing Files ({len(self.completed_items)} / {total_files_count})", False),
            refresh=False,
        )
        self.data_processing_bar.set_postfix_str(
            _format_speed_postfix(group_report.total_bytes_completion_rate), refresh=False
        )
        self.data_processing_bar.update(bytes_inc)

        self.upload_bar.total = group_report.total_transfer_bytes
        self.upload_bar.set_postfix_str(
            _format_speed_postfix(group_report.total_transfer_bytes_completion_rate), refresh=False
        )
        self.upload_bar.update(transfer_inc)

        # Single Validating bar for shard finalization (v2 shard API).
        # Prefer validation-entry counts (the long phase). Fall back to shard-completion
        # counts when entries are absent (e.g. V1 synthetic Result with no Validating frames).
        # Set absolute n/total for the current commit; reset_for_next_commit restarts at 0%.
        shard = getattr(group_report, "shard", None)
        if shard is not None and shard.total_shards > 0:
            if self.validating_bar is None:
                self._init_validating_bar()

            assert self.validating_bar is not None

            if shard.total_shard_validation_entries > 0:
                done_n, total_n = (
                    shard.total_shard_validation_entries_completed,
                    shard.total_shard_validation_entries,
                )
            else:
                done_n, total_n = shard.total_shards_completed, shard.total_shards

            self.validating_bar.total = total_n or None
            self.validating_bar.n = done_n
            # Keep Validating snug under the active file bars as more slots fill in.
            self.validating_bar.pos = -self._validating_bar_position()
            self.validating_bar.refresh()

    def close(self):
        self.data_processing_bar.close()
        self.upload_bar.close()

        if self.validating_bar is not None:
            self.validating_bar.close()

        if self.per_file_progress:
            for bar in self.current_bars:
                if bar:
                    bar.close()
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
"""Contains utilities to handle HTTP requests in huggingface_hub."""

import atexit
import io
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from shlex import quote
from typing import Any, TypeVar
from urllib.parse import urljoin, urlparse

import httpx2

from huggingface_hub.errors import OfflineModeIsEnabled

from .. import constants
from ..errors import (
    BadRequestError,
    BucketNotFoundError,
    DisabledRepoError,
    GatedRepoError,
    HfHubHTTPError,
    JobNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from . import logging
from ._lfs import SliceFileObj
from ._typing import HTTP_METHOD_T, CallableT


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class RateLimitInfo:
    """
    Parsed rate limit information from HTTP response headers.

    Attributes:
        resource_type (`str`): The type of resource being rate limited.
        remaining (`int`): The number of requests remaining in the current window.
        reset_in_seconds (`int`): The number of seconds until the rate limit resets.
        limit (`int`, *optional*): The maximum number of requests allowed in the current window.
        window_seconds (`int`, *optional*): The number of seconds in the current window.

    """

    resource_type: str
    remaining: int
    reset_in_seconds: int
    limit: int | None = None
    window_seconds: int | None = None


# Regex patterns for parsing rate limit headers
# e.g.: "api";r=0;t=55 --> resource_type="api", r=0, t=55
_RATELIMIT_REGEX = re.compile(r"\"(?P<resource_type>\w+)\"\s*;\s*r\s*=\s*(?P<r>\d+)\s*;\s*t\s*=\s*(?P<t>\d+)")
# e.g.: "fixed window";"api";q=500;w=300 --> q=500, w=300
_RATELIMIT_POLICY_REGEX = re.compile(r"q\s*=\s*(?P<q>\d+).*?w\s*=\s*(?P<w>\d+)")


def parse_ratelimit_headers(headers: Mapping[str, str]) -> RateLimitInfo | None:
    """Parse rate limit information from HTTP response headers.

    Follows IETF draft: https://www.ietf.org/archive/id/draft-ietf-httpapi-ratelimit-headers-09.html
    Only a subset is implemented.

    Example:
    ```python
    >>> from huggingface_hub.utils import parse_ratelimit_headers
    >>> headers = {
    ...     "ratelimit": '"api";r=0;t=55',
    ...     "ratelimit-policy": '"fixed window";"api";q=500;w=300',
    ... }
    >>> info = parse_ratelimit_headers(headers)
    >>> info.remaining
    0
    >>> info.reset_in_seconds
    55
    ```
    """

    ratelimit: str | None = None
    policy: str | None = None
    for key in headers:
        lower_key = key.lower()
        if lower_key == "ratelimit":
            ratelimit = headers[key]
        elif lower_key == "ratelimit-policy":
            policy = headers[key]

    if not ratelimit:
        return None

    match = _RATELIMIT_REGEX.search(ratelimit)
    if not match:
        return None

    resource_type = match.group("resource_type")
    remaining = int(match.group("r"))
    reset_in_seconds = int(match.group("t"))

    limit: int | None = None
    window_seconds: int | None = None

    if policy:
        policy_match = _RATELIMIT_POLICY_REGEX.search(policy)
        if policy_match:
            limit = int(policy_match.group("q"))
            window_seconds = int(policy_match.group("w"))

    return RateLimitInfo(
        resource_type=resource_type,
        remaining=remaining,
        reset_in_seconds=reset_in_seconds,
        limit=limit,
        window_seconds=window_seconds,
    )


def _parse_retry_after(headers: Mapping[str, str]) -> int | None:
    """Parse the standard `Retry-After` HTTP header into a number of seconds to wait.

    The `Retry-After` header can be either a non-negative number of seconds (delay-seconds)
    or an HTTP-date after which to retry. We handle only the delay-seconds case.

    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After.
    """
    value: str | None = None
    for key in headers:
        if key.lower() == "retry-after":
            value = headers[key]
            break

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    if value.isdigit():
        return int(value)  # e.g. "Retry-After: 120"
    return None  #  e.g. "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT" - not supported


# When raising an error, we include the request id in the error message for easier debugging.
# Request ID is sourced from headers in order of precedence: "X-Request-Id", "X-Amzn-Trace-Id", "X-Amz-Cf-Id".
X_REQUEST_ID = "x-request-id"
X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"
X_AMZ_CF_ID = "x-amz-cf-id"

REPO_API_REGEX = re.compile(
    r"""
        # staging or production endpoint
        ^https://[^/]+
        (
            # on /api/repo_type/repo_id
            /api/(models|datasets|spaces)/(.+)
            |
            # or /repo_id/resolve/revision/...
            /(.+)/resolve/(.+)
        )
    """,
    flags=re.VERBOSE,
)

BUCKET_API_REGEX = re.compile(
    r"""
        # staging or production endpoint
        ^https?://[^/]+
        # on /api/buckets/...
        /api/buckets/
    """,
    flags=re.VERBOSE,
)

# Regex to extract the job_id from a (scheduled) job API URL.
# Matches /api/jobs/{namespace}/{job_id}[/...] and /api/scheduled-jobs/{namespace}/{job_id}[/...].
_JOB_ID_FROM_URL_REGEX = re.compile(r"^https?://[^/]+/api/(?:scheduled-jobs|jobs)/[^/]+/([^/?]+)")

# Regex to extract repo_type and repo_id from API URLs.
# Captures: group(1) = repo_type plural (models/datasets/spaces), group(2) = first path segment, group(3) = optional second segment.
_REPO_ID_FROM_URL_REGEX = re.compile(r"^https?://[^/]+/api/(models|datasets|spaces)/([^/?]+)(?:/([^/?]+))?")

# Regex to extract repo_type and repo_id from download URLs: /[{repo_type}s/]{repo_id}/resolve/...
# Captures: group(1) = optional repo_type plural (datasets/spaces/kernels), group(2) = repo_id.
_REPO_ID_FROM_RESOLVE_URL_REGEX = re.compile(
    r"^https?://[^/]+/(?:(datasets|spaces|kernels)/)?([^/?]+(?:/[^/?]+)?)/resolve/"
)

# Regex to extract bucket_id (namespace/name) from bucket API URLs.
_BUCKET_ID_FROM_URL_REGEX = re.compile(r"^https?://[^/]+/api/buckets/([^/?]+/[^/?]+)")

# Sub-paths that follow a repo_id in API URLs (not part of the repo name).
_REPO_URL_SUBPATHS = {"resolve", "tree", "blob", "raw", "refs", "commit", "discussions", "settings", "revision"}


def _parse_repo_info_from_url(url: str) -> tuple[str | None, str | None]:
    """Extract (repo_type, repo_id) from an API URL or a download URL.

    Returns canonical repo_type values: "model", "dataset", "space" (or None).

    Examples:
        >>> _parse_repo_info_from_url("https://huggingface.co/api/models/user/repo")
        ("model", "user/repo")
        >>> _parse_repo_info_from_url("https://huggingface.co/api/datasets/user/repo/resolve/main/data.csv")
        ("dataset", "user/repo")
        >>> _parse_repo_info_from_url("https://huggingface.co/api/models/bert-base-cased/resolve/main/config.json")
        ("model", "bert-base-cased")
        >>> _parse_repo_info_from_url("https://huggingface.co/datasets/user/repo/resolve/main/data.csv")
        ("dataset", "user/repo")
    """
    if match := _REPO_ID_FROM_RESOLVE_URL_REGEX.search(url):
        return constants.REPO_TYPES_MAPPING[match.group(1) or "models"], match.group(2)
    match = _REPO_ID_FROM_URL_REGEX.search(url)
    if not match:
        return None, None
    repo_type = constants.REPO_TYPES_MAPPING.get(match.group(1))
    first, second = match.group(2), match.group(3)
    if second and second not in _REPO_URL_SUBPATHS:
        repo_id = f"{first}/{second}"
    else:
        repo_id = first
    return repo_type, repo_id


def _parse_bucket_id_from_url(url: str) -> str | None:
    """Extract bucket_id (namespace/name) from a bucket API URL."""
    match = _BUCKET_ID_FROM_URL_REGEX.search(url)
    return match.group(1) if match else None


def _parse_job_id_from_url(url: str) -> str | None:
    """Extract the job_id from a (scheduled) job API URL, if present."""
    match = _JOB_ID_FROM_URL_REGEX.search(url)
    return match.group(1) if match else None


_IS_DOWNLOAD_CALL: ContextVar[bool] = ContextVar("_IS_DOWNLOAD_CALL", default=False)


def flag_as_download_call(fn: CallableT) -> CallableT:
    """Decorator to tag all HTTP calls made by `fn` with the `X-HF-Download-Counter` header.

    Applied on `hf_hub_download`, `snapshot_download`, etc. so that every call they make through [`get_session`]
    is flagged as part of a download, no matter how deep in the call stack it happens.
    """

    @wraps(fn)
    def _inner(*args, **kwargs):
        token = _IS_DOWNLOAD_CALL.set(True)
        try:
            return fn(*args, **kwargs)
        finally:
            _IS_DOWNLOAD_CALL.reset(token)

    return _inner  # type: ignore


def hf_request_event_hook(request: httpx2.Request) -> str | None:
    """
    Event hook that will be used to make HTTP requests to the Hugging Face Hub.

    What it does:
    - Block requests if offline mode is enabled
    - Add a request ID to the request headers
    - Flag the request as part of a download, if applicable
    - Log the request if debug mode is enabled
    """
    if constants.is_offline_mode():
        raise OfflineModeIsEnabled(
            f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
        )

    if _IS_DOWNLOAD_CALL.get():
        request.headers[constants.X_HF_DOWNLOAD_COUNTER] = "1"

    # Add random request ID => easier for server-side debugging
    if X_AMZN_TRACE_ID not in request.headers:
        request.headers[X_AMZN_TRACE_ID] = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())
    request_id = request.headers.get(X_AMZN_TRACE_ID)

    # Debug log
    logger.debug(
        "Request %s: %s %s (authenticated: %s)",
        request_id,
        request.method,
        request.url,
        request.headers.get("authorization") is not None,
    )
    if constants.HF_DEBUG:
        logger.debug("Send: %s", _curlify(request))

    return request_id


async def async_hf_request_event_hook(request: httpx2.Request) -> str | None:
    """
    Async version of `hf_request_event_hook`.
    """
    return hf_request_event_hook(request)


async def async_hf_response_event_hook(response: httpx2.Response) -> None:
    if response.status_code >= 400:
        # If response will raise, read content from stream to have it available when raising the exception
        # If content-length is not set or is too large, skip reading the content to avoid OOM
        if "Content-length" in response.headers:
            try:
                length = int(response.headers["Content-length"])
            except ValueError:
                return

            if length < 1_000_000:
                await response.aread()


def default_client_factory() -> httpx2.Client:
    """
    Factory function to create a `httpx2.Client` with the default transport.
    """
    return httpx2.Client(
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    )


def default_async_client_factory() -> httpx2.AsyncClient:
    """
    Factory function to create a `httpx2.AsyncClient` with the default transport.
    """
    return httpx2.AsyncClient(
        event_hooks={"request": [async_hf_request_event_hook], "response": [async_hf_response_event_hook]},
        follow_redirects=True,
        timeout=None,
    )


CLIENT_FACTORY_T = Callable[[], httpx2.Client]
ASYNC_CLIENT_FACTORY_T = Callable[[], httpx2.AsyncClient]

_CLIENT_LOCK = threading.Lock()
_GLOBAL_CLIENT_FACTORY: CLIENT_FACTORY_T = default_client_factory
_GLOBAL_ASYNC_CLIENT_FACTORY: ASYNC_CLIENT_FACTORY_T = default_async_client_factory
_GLOBAL_CLIENT: httpx2.Client | None = None


def set_client_factory(client_factory: CLIENT_FACTORY_T) -> None:
    """
    Set the HTTP client factory to be used by `huggingface_hub`.

    The client factory is a method that returns a `httpx2.Client` object. On the first call to [`get_session`] the client factory
    will be used to create a new `httpx2.Client` object that will be shared between all calls made by `huggingface_hub`.

    This can be useful if you are running your scripts in a specific environment requiring custom configuration (e.g. custom proxy or certifications).

    Use [`get_session`] to get a correctly configured `httpx2.Client`.
    """
    global _GLOBAL_CLIENT_FACTORY
    with _CLIENT_LOCK:
        close_session()
        _GLOBAL_CLIENT_FACTORY = client_factory


def set_async_client_factory(async_client_factory: ASYNC_CLIENT_FACTORY_T) -> None:
    """
    Set the HTTP async client factory to be used by `huggingface_hub`.

    The async client factory is a method that returns a `httpx2.AsyncClient` object.
    This can be useful if you are running your scripts in a specific environment requiring custom configuration (e.g. custom proxy or certifications).
    Use [`get_async_session`] to get a correctly configured `httpx2.AsyncClient`.

    > [!WARNING]
    > Contrary to the `httpx2.Client` that is shared between all calls made by `huggingface_hub`, the `httpx2.AsyncClient` is not shared.
    > It is recommended to use an async context manager to ensure the client is properly closed when the context is exited.
    """
    global _GLOBAL_ASYNC_CLIENT_FACTORY
    _GLOBAL_ASYNC_CLIENT_FACTORY = async_client_factory


def get_session() -> httpx2.Client:
    """
    Get a `httpx2.Client` object, using the transport factory from the user.

    This client is shared between all calls made by `huggingface_hub`. Therefore you should not close it manually.

    Use [`set_client_factory`] to customize the `httpx2.Client`.
    """
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        with _CLIENT_LOCK:
            if _GLOBAL_CLIENT is None:
                _GLOBAL_CLIENT = _GLOBAL_CLIENT_FACTORY()
    return _GLOBAL_CLIENT


def get_async_session() -> httpx2.AsyncClient:
    """
    Return a `httpx2.AsyncClient` object, using the transport factory from the user.

    Use [`set_async_client_factory`] to customize the `httpx2.AsyncClient`.

    > [!WARNING]
    > Contrary to the `httpx2.Client` that is shared between all calls made by `huggingface_hub`, the `httpx2.AsyncClient` is not shared.
    > It is recommended to use an async context manager to ensure the client is properly closed when the context is exited.
    """
    return _GLOBAL_ASYNC_CLIENT_FACTORY()


def close_session() -> None:
    """
    Close the global `httpx2.Client` used by `huggingface_hub`.

    If a Client is closed, it will be recreated on the next call to [`get_session`].

    Can be useful if e.g. an SSL certificate has been updated.
    """
    global _GLOBAL_CLIENT
    client = _GLOBAL_CLIENT

    # First, set global client to None
    _GLOBAL_CLIENT = None

    # Then, close the clients
    if client is not None:
        try:
            client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")


atexit.register(close_session)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=close_session)


_DEFAULT_RETRY_ON_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx2.TimeoutException,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
)
_DEFAULT_RETRY_ON_STATUS_CODES: tuple[int, ...] = (408, 429, 500, 502, 503, 504)


def _http_backoff_base(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: type[Exception] | tuple[type[Exception], ...] = _DEFAULT_RETRY_ON_EXCEPTIONS,
    retry_on_status_codes: int | tuple[int, ...] = _DEFAULT_RETRY_ON_STATUS_CODES,
    stream: bool = False,
    **kwargs,
) -> Generator[httpx2.Response, None, None]:
    """Internal implementation of HTTP backoff logic shared between `http_backoff` and `http_stream_backoff`."""
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    nb_tries = 0
    sleep_time = base_wait_time
    ratelimit_reset: int | None = None  # seconds to wait for rate limit reset if 429 response

    # If `data` is used and is a file object (or any IO), it will be consumed on the
    # first HTTP request. We need to save the initial position so that the full content
    # of the file is re-sent on http backoff. See warning tip in docstring.
    io_obj_initial_pos = None
    if "data" in kwargs and isinstance(kwargs["data"], (io.IOBase, SliceFileObj)):
        io_obj_initial_pos = kwargs["data"].tell()

    while True:
        nb_tries += 1
        ratelimit_reset = None
        # Fetched on each attempt: a previous attempt may have closed the shared client (see `close_session` below),
        # and closed `httpx2.Client` objects cannot be reused.
        client = get_session()
        try:
            # If `data` is used and is a file object (or any IO), set back cursor to
            # initial position.
            if io_obj_initial_pos is not None:
                kwargs["data"].seek(io_obj_initial_pos)

            # Perform request and handle response
            def _should_retry(response: httpx2.Response) -> bool:
                """Handle response and return True if should retry, False if should return/yield."""
                nonlocal ratelimit_reset

                if response.status_code not in retry_on_status_codes:
                    return False  # Success, don't retry

                # Wrong status code returned (HTTP 503 for instance)
                logger.warning(f"HTTP Error {response.status_code} thrown while requesting {method} {url}")
                if nb_tries > max_retries:
                    hf_raise_for_status(response)  # Will raise uncaught exception
                    # Return/yield response to avoid infinite loop in the corner case where the
                    # user ask for retry on a status code that doesn't raise_for_status.
                    return False  # Don't retry, return/yield response

                # Check 'ratelimit' and `Retry-After` headers.
                if (
                    response.status_code == 429
                    and (ratelimit_info := parse_ratelimit_headers(response.headers)) is not None
                    and ratelimit_info.remaining == 0
                ):
                    ratelimit_reset = ratelimit_info.reset_in_seconds
                elif (retry_after := _parse_retry_after(response.headers)) is not None:
                    ratelimit_reset = retry_after

                return True  # Should retry

            if stream:
                with client.stream(method=method, url=url, **kwargs) as response:
                    if not _should_retry(response):
                        yield response
                        return
            else:
                response = client.request(method=method, url=url, **kwargs)
                if not _should_retry(response):
                    yield response
                    return

        except retry_on_exceptions as err:
            logger.warning(f"'{err}' thrown while requesting {method} {url}")

            if isinstance(err, httpx2.ConnectError):
                close_session()  # In case of SSLError it's best to close the shared httpx2.Client objects

            if nb_tries > max_retries:
                raise err

        if ratelimit_reset is not None:
            actual_sleep = float(ratelimit_reset) + 1  # +1s to avoid rounding issues
            logger.warning(f"Rate limited. Waiting {actual_sleep}s before retry [Retry {nb_tries}/{max_retries}].")
        else:
            actual_sleep = sleep_time
            logger.warning(f"Retrying in {actual_sleep}s [Retry {nb_tries}/{max_retries}].")

        time.sleep(actual_sleep)

        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff


def http_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: type[Exception] | tuple[type[Exception], ...] = _DEFAULT_RETRY_ON_EXCEPTIONS,
    retry_on_status_codes: int | tuple[int, ...] = _DEFAULT_RETRY_ON_STATUS_CODES,
    **kwargs,
) -> httpx2.Response:
    """Wrapper around httpx2 to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5. Set to `0` to disable retries.
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`type[Exception]` or `tuple[type[Exception]]`, *optional*):
            Define which exceptions must be caught to retry the request. Can be a single type or a tuple of types.
            By default, retry on `httpx2.TimeoutException`, `httpx2.NetworkError` and `httpx2.RemoteProtocolError`.
        retry_on_status_codes (`int` or `tuple[int]`, *optional*, defaults to `(429, 500, 502, 503, 504)`):
            Define on which status codes the request must be retried. By default, retries
            on rate limit (429) and server errors (5xx).
        **kwargs (`dict`, *optional*):
            kwargs to pass to `httpx2.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_backoff

    # Same usage as "httpx2.request".
    >>> response = http_backoff("GET", "https://www.google.com")
    >>> response.raise_for_status()

    # If you expect a Gateway Timeout from time to time
    >>> http_backoff("PUT", upload_url, data=data, retry_on_status_codes=504)
    >>> response.raise_for_status()
    ```

    > [!WARNING]
    > When using `requests` it is possible to stream data by passing an iterator to the
    > `data` argument. On http backoff this is a problem as the iterator is not reset
    > after a failed call. This issue is mitigated for file objects or any IO streams
    > by saving the initial position of the cursor (with `data.tell()`) and resetting the
    > cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    > will fail. If this is a hard constraint for you, please let us know by opening an
    > issue on [Github](https://github.com/huggingface/huggingface_hub).
    """
    return next(
        _http_backoff_base(
            method=method,
            url=url,
            max_retries=max_retries,
            base_wait_time=base_wait_time,
            max_wait_time=max_wait_time,
            retry_on_exceptions=retry_on_exceptions,
            retry_on_status_codes=retry_on_status_codes,
            stream=False,
            **kwargs,
        )
    )


@contextmanager
def http_stream_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: type[Exception] | tuple[type[Exception], ...] = _DEFAULT_RETRY_ON_EXCEPTIONS,
    retry_on_status_codes: int | tuple[int, ...] = _DEFAULT_RETRY_ON_STATUS_CODES,
    **kwargs,
) -> Generator[httpx2.Response, None, None]:
    """Wrapper around httpx2 to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5. Set to `0` to disable retries.
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`type[Exception]` or `tuple[type[Exception]]`, *optional*):
            Define which exceptions must be caught to retry the request. Can be a single type or a tuple of types.
            By default, retry on `httpx2.TimeoutException`, `httpx2.NetworkError` and `httpx2.RemoteProtocolError`.
        retry_on_status_codes (`int` or `tuple[int]`, *optional*, defaults to `(429, 500, 502, 503, 504)`):
            Define on which status codes the request must be retried. By default, retries
            on rate limit (429) and server errors (5xx).
        **kwargs (`dict`, *optional*):
            kwargs to pass to `httpx2.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_stream_backoff

    # Same usage as "httpx2.stream".
    >>> with http_stream_backoff("GET", "https://www.google.com") as response:
    ...     for chunk in response.iter_bytes():
    ...         print(chunk)

    # If you expect a Gateway Timeout from time to time
    >>> with http_stream_backoff("PUT", upload_url, data=data, retry_on_status_codes=504) as response:
    ...     response.raise_for_status()
    ```

    > [!WARNING]
    > When using `httpx2` it is possible to stream data by passing an iterator to the
    > `data` argument. On http backoff this is a problem as the iterator is not reset
    > after a failed call. This issue is mitigated for file objects or any IO streams
    > by saving the initial position of the cursor (with `data.tell()`) and resetting the
    > cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    > will fail. If this is a hard constraint for you, please let us know by opening an
    > issue on [Github](https://github.com/huggingface/huggingface_hub).
    """
    yield from _http_backoff_base(
        method=method,
        url=url,
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        retry_on_exceptions=retry_on_exceptions,
        retry_on_status_codes=retry_on_status_codes,
        stream=True,
        **kwargs,
    )


_MAX_REDIRECTS = 20  # same bound as httpx2's default max_redirects


def _httpx2_follow_hub_redirects_with_backoff(
    method: HTTP_METHOD_T, url: str, *, retry_on_errors: bool = False, **httpx2_kwargs
) -> httpx2.Response:
    """Perform an HTTP request with backoff, following redirects that stay on the Hub.

    Used to fetch HEAD /resolve on repo or bucket files.

    Redirects to the same host or another Hub host are followed. Redirects to any other host (CDN, storage
    bucket) are not: the file metadata is on the redirect response itself and the auth header must not leave the Hub.

    A backoff mechanism retries the HTTP call on errors (429, 5xx, timeout, network errors).

    Args:
        method (`str`):
            HTTP method, such as 'GET' or 'HEAD'.
        url (`str`):
            The URL of the resource to fetch.
        retry_on_errors (`bool`, *optional*, defaults to `False`):
            Whether to retry on errors. If False, no retry is performed (fast fallback to local cache).
            If True, uses default retry behavior (429, 5xx, timeout, network errors).
        **httpx2_kwargs (`dict`, *optional*):
            Params to pass to `httpx2.request`.
    """
    # if `retry_on_errors=False`, disable all retries for fast fallback to cache
    no_retry_kwargs: dict[str, Any] = (
        {} if retry_on_errors else {"retry_on_exceptions": (), "retry_on_status_codes": ()}
    )

    for _ in range(_MAX_REDIRECTS):
        response = http_backoff(
            method=method,
            url=url,
            **httpx2_kwargs,
            follow_redirects=False,
            **no_retry_kwargs,
        )
        hf_raise_for_status(response)

        if not response.has_redirect_location:
            return response

        target = urljoin(url, response.headers["Location"])
        if not _is_same_or_hub_host(url, target):
            # Redirect to a CDN or storage host (signed URL): the file metadata is carried by this very response,
            # and the authorization header must not be forwarded off the Hub => stop here.
            return response

        url = target

    raise httpx2.TooManyRedirects(f"Exceeded {_MAX_REDIRECTS} redirects while resolving '{url}'.")


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_origin(url: str) -> tuple[str, str, int]:
    """Origin of a url: (scheme, hostname, port), with the scheme's default port if not explicit."""
    parsed = urlparse(url)
    return (parsed.scheme, (parsed.hostname or "").lower(), parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 0))


# Origins considered to be Hub endpoints. The auth header is forwarded to these: only add trusted ones.
_HUB_ORIGINS = frozenset(
    _url_origin(endpoint)
    for endpoint in (
        constants._HF_DEFAULT_ENDPOINT,
        constants._HF_DEFAULT_STAGING_ENDPOINT,
        constants.ENDPOINT,
        "https://hf.co",
    )
)


def _is_same_or_hub_host(url: str, target: str) -> bool:
    """Whether `target` is served by the same origin as `url`, or by a known Hub origin."""
    return _url_origin(target) in {_url_origin(url), *_HUB_ORIGINS}


def fix_hf_endpoint_in_url(url: str, endpoint: str | None) -> str:
    """Replace the default endpoint in a URL by a custom one.

    This is useful when using a proxy and the Hugging Face Hub returns a URL with the default endpoint.
    """
    endpoint = endpoint.rstrip("/") if endpoint else constants.ENDPOINT
    # check if a proxy has been set => if yes, update the returned URL to use the proxy
    if endpoint not in (constants._HF_DEFAULT_ENDPOINT, constants._HF_DEFAULT_STAGING_ENDPOINT):
        url = url.replace(constants._HF_DEFAULT_ENDPOINT, endpoint)
        url = url.replace(constants._HF_DEFAULT_STAGING_ENDPOINT, endpoint)
    return url


def hf_raise_for_status(response: httpx2.Response, endpoint_name: str | None = None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a potential HTTPError.
    Raised exception will be an instance of [`~errors.HfHubHTTPError`].

    This helper is meant to be the unique method to raise_for_status when making a call to the Hugging Face Hub.

    Args:
        response (`Response`):
            Response from the server.
        endpoint_name (`str`, *optional*):
            Name of the endpoint that has been called. If provided, the error message will be more complete.

    > [!WARNING]
    > Raises when the request has failed:
    >
    >     - [`~utils.RepositoryNotFoundError`]
    >         If the repository to download from cannot be found. This may be because it
    >         doesn't exist, because `repo_type` is not set correctly, or because the repo
    >         is `private` and you do not have access.
    >     - [`~utils.GatedRepoError`]
    >         If the repository exists but is gated and the user is not on the authorized
    >         list.
    >     - [`~utils.RevisionNotFoundError`]
    >         If the repository exists but the revision couldn't be found.
    >     - [`~utils.EntryNotFoundError`]
    >         If the repository exists but the entry (e.g. the requested file) couldn't be
    >         find.
    >     - [`~utils.BadRequestError`]
    >         If request failed with a HTTP 400 BadRequest error.
    >     - [`~utils.HfHubHTTPError`]
    >         If request failed for a reason not listed above.
    """
    try:
        _warn_on_warning_headers(response)
    except Exception:
        # Never raise on warning parsing
        logger.debug("Failed to parse warning headers", exc_info=True)

    try:
        response.raise_for_status()
    except httpx2.HTTPStatusError as e:
        if response.status_code // 100 == 3:
            return  # Do not raise on redirects to stay consistent with `requests`

        error_code = response.headers.get("X-Error-Code")
        error_message = response.headers.get("X-Error-Message")

        # Parse repo info from request URL (used to enrich errors below)
        request_url = (
            str(response.request.url) if response.request is not None and response.request.url is not None else None
        )
        repo_type, repo_id = _parse_repo_info_from_url(request_url) if request_url else (None, None)

        if error_code == "RevisionNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Revision Not Found for url: {response.url}."
            raise _format(RevisionNotFoundError, message, response, repo_type=repo_type, repo_id=repo_id) from e

        elif error_code == "EntryNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise _format(RemoteEntryNotFoundError, message, response, repo_type=repo_type, repo_id=repo_id) from e

        elif error_code == "GatedRepo":
            message = (
                f"{response.status_code} Client Error." + "\n\n" + f"Cannot access gated repo for url {response.url}."
            )
            raise _format(GatedRepoError, message, response, repo_type=repo_type, repo_id=repo_id) from e

        elif error_message == "Access to this resource is disabled.":
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Cannot access repository for url {response.url}."
                + "\n"
                + "Access to this resource is disabled."
            )
            raise _format(DisabledRepoError, message, response) from e

        elif (
            error_code == "RepoNotFound"
            and request_url is not None
            and BUCKET_API_REGEX.search(request_url) is not None
        ):
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Bucket Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct bucket id (namespace/name)."
                + "\nIf the bucket is private, make sure you are authenticated and your token has the required permissions."
            )
            raise _format(
                BucketNotFoundError, message, response, bucket_id=_parse_bucket_id_from_url(request_url)
            ) from e

        elif (
            response.status_code == 404
            and request_url is not None
            and (job_id := _parse_job_id_from_url(request_url)) is not None
        ):
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Job Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct job ID and namespace."
            )
            raise _format(JobNotFoundError, message, response, job_id=job_id) from e

        elif error_code == "RepoNotFound" or (
            response.status_code == 401
            and error_message != "Invalid credentials in Authorization header"
            and request_url is not None
            and REPO_API_REGEX.search(request_url) is not None
        ):
            # 401 is misleading as it is returned for:
            #    - private and gated repos if user is not authenticated
            #    - missing repos
            # => for now, we process them as `RepoNotFound` anyway.
            # See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Repository Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct `repo_id` and"
                " `repo_type`.\nIf you are trying to access a private or gated repo,"
                " make sure you are authenticated and your token has the required permissions."
                + "\nFor more details, see https://huggingface.co/docs/huggingface_hub/authentication"
            )
            raise _format(RepositoryNotFoundError, message, response, repo_type=repo_type, repo_id=repo_id) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise _format(BadRequestError, message, response) from e

        elif response.status_code == 403:
            message = (
                f"\n\n{response.status_code} Forbidden: {error_message}."
                + f"\nCannot access content at: {response.url}."
                + "\nMake sure your token has the correct permissions."
            )
            raise _format(HfHubHTTPError, message, response) from e

        elif response.status_code == 429:
            ratelimit_info = parse_ratelimit_headers(response.headers)
            # Headers are set on every response: only use them if they are the reason for the 429.
            if ratelimit_info is not None and ratelimit_info.remaining == 0:
                message = (
                    f"\n\n429 Too Many Requests: you have reached your '{ratelimit_info.resource_type}' rate limit."
                )
                message += f"\nRetry after {ratelimit_info.reset_in_seconds} seconds"
                if ratelimit_info.limit is not None and ratelimit_info.window_seconds is not None:
                    message += (
                        f" ({ratelimit_info.remaining}/{ratelimit_info.limit} requests remaining"
                        f" in current {ratelimit_info.window_seconds}s window)."
                    )
                else:
                    message += "."
                message += f"\nUrl: {response.url}."
            else:
                message = f"\n\n429 Too Many Requests for url: {response.url}."
            raise _format(HfHubHTTPError, message, response) from e

        elif response.status_code == 416:
            range_header = response.request.headers.get("Range")
            message = f"{e}. Requested range: {range_header}. Content-Range: {response.headers.get('Content-Range')}."
            raise _format(HfHubHTTPError, message, response) from e

        # Convert `HTTPError` into a `HfHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise _format(HfHubHTTPError, str(e), response) from e


_WARNED_TOPICS = set()


def _warn_on_warning_headers(response: httpx2.Response) -> None:
    """
    Emit warnings if warning headers are present in the HTTP response.

    Expected header format: 'X-HF-Warning: topic; message'

    Only the first warning for each topic will be shown. Topic is optional and can be empty. Note that several warning
    headers can be present in a single response.

    Args:
        response (`httpx2.Response`):
            The HTTP response to check for warning headers.
    """
    server_warnings = response.headers.get_list("X-HF-Warning")
    for server_warning in server_warnings:
        topic, message = server_warning.split(";", 1) if ";" in server_warning else ("", server_warning)
        topic = topic.strip()
        if topic not in _WARNED_TOPICS:
            message = message.strip()
            if message:
                _WARNED_TOPICS.add(topic)
                logger.warning(message)


_HfHubHTTPErrorT = TypeVar("_HfHubHTTPErrorT", bound=HfHubHTTPError)


def _format(
    error_type: type[_HfHubHTTPErrorT], custom_message: str, response: httpx2.Response, **attrs: Any
) -> _HfHubHTTPErrorT:
    server_errors = []

    # Retrieve server error from header
    from_headers = response.headers.get("X-Error-Message")
    if from_headers is not None:
        server_errors.append(from_headers)

    # Retrieve server error from body
    try:
        # Case errors are returned in a JSON format
        try:
            data = response.json()
        except httpx2.ResponseNotRead:
            try:
                response.read()  # In case of streaming response, we need to read the response first
                data = response.json()
            except RuntimeError:
                # In case of async streaming response, we can't read the stream here.
                # In practice if user is using the default async client from `get_async_client`, the stream will have
                # already been read in the async event hook `async_hf_response_event_hook`.
                #
                # Here, we are skipping reading the response to avoid RuntimeError but it happens only if async + stream + used httpx2.AsyncClient directly.
                data = {}

        error = data.get("error")
        error_description = data.get("error_description")
        if error is not None:
            if isinstance(error, list):
                # Case {'error': ['my error 1', 'my error 2']}
                server_errors.extend(error)
            elif error_description is not None:
                # OAuth-style case {'error': 'invalid_grant', 'error_description': 'my description'}
                server_errors.append(f"{error}: {error_description}")
            else:
                # Case {'error': 'my error'}
                server_errors.append(error)
        elif error_description is not None:
            # Case {'error_description': 'my description'} (no 'error' field)
            server_errors.append(error_description)

        errors = data.get("errors")
        if errors is not None:
            # Case {'errors': [{'message': 'my error 1'}, {'message': 'my error 2'}]}
            for error in errors:
                if "message" in error:
                    server_errors.append(error["message"])

    except json.JSONDecodeError:
        # If content is not JSON and not HTML, append the text
        content_type = response.headers.get("Content-Type", "")
        if response.text and "html" not in content_type.lower():
            server_errors.append(response.text)

    # Strip all server messages
    server_errors = [str(line).strip() for line in server_errors if str(line).strip()]

    # Deduplicate server messages (keep order)
    # taken from https://stackoverflow.com/a/17016257
    server_errors = list(dict.fromkeys(server_errors))

    # Format server error
    server_message = "\n".join(server_errors)

    # Add server error to custom message
    final_error_message = custom_message
    if server_message and server_message.lower() not in custom_message.lower():
        if "\n\n" in custom_message:
            final_error_message += "\n" + server_message
        else:
            final_error_message += "\n\n" + server_message

    # Prepare Request ID message
    request_id = ""
    request_id_message = ""
    for header, label in (
        (X_REQUEST_ID, "Request ID"),
        (X_AMZN_TRACE_ID, "Amzn Trace ID"),
        (X_AMZ_CF_ID, "Amz CF ID"),
    ):
        value = response.headers.get(header)
        if value:
            request_id = str(value)
            request_id_message = f" ({label}: {value})"
            break

    # Add Request ID
    if request_id and request_id.lower() not in final_error_message.lower():
        if "\n" in final_error_message:
            newline_index = final_error_message.index("\n")
            final_error_message = (
                final_error_message[:newline_index] + request_id_message + final_error_message[newline_index:]
            )
        else:
            final_error_message += request_id_message

    # Return
    err = error_type(final_error_message.strip(), response=response, server_message=server_message or None)
    for k, v in attrs.items():
        setattr(err, k, v)
    return err


# Request-body fields that carry credentials; redacted from debug-log curl commands (HF_DEBUG).
_SENSITIVE_BODY_KEYS = ("subject_token", "access_token", "refresh_token", "client_secret", "device_code")
_SENSITIVE_BODY_PATTERNS = [
    pattern
    for key in _SENSITIVE_BODY_KEYS
    for pattern in (
        (re.compile(rf'("{key}"\s*:\s*")[^"]*(")'), r"\1<REDACTED>\2"),  # JSON
        (re.compile(rf"(^|&)({key}=)[^&]*"), r"\1\2<REDACTED>"),  # application/x-www-form-urlencoded
    )
]


def _redact_sensitive_body(body: str) -> str:
    """Redact OAuth credential values from a JSON or form-urlencoded request body string."""
    for pattern, replacement in _SENSITIVE_BODY_PATTERNS:
        body = pattern.sub(replacement, body)
    return body


def _curlify(request: httpx2.Request) -> str:
    """Convert a `httpx2.Request` into a curl command (str).

    Used for debug purposes only.

    Implementation vendored from https://github.com/ofw/curlify/blob/master/curlify.py.
    MIT License Copyright (c) 2016 Egor.
    """
    parts: list[tuple[Any, Any]] = [
        ("curl", None),
        ("-X", request.method),
    ]

    for k, v in sorted(request.headers.items()):
        if k.lower() == "authorization":
            v = "<TOKEN>"  # Hide authorization header, no matter its value (can be Bearer, Key, etc.)
        parts += [("-H", f"{k}: {v}")]

    body: str | None = None
    try:
        if request.content is not None:
            body = request.content.decode("utf-8", errors="ignore")
            if len(body) > 1000:
                body = f"{body[:1000]} ... [truncated]"
            body = _redact_sensitive_body(body)
    except httpx2.RequestNotRead:
        body = "<streaming body>"
    if body is not None:
        parts += [("-d", body.replace("\n", ""))]

    parts += [(None, request.url)]

    flat_parts = []
    for k, v in parts:
        if k:
            flat_parts.append(quote(str(k)))
        if v:
            flat_parts.append(quote(str(v)))

    return " ".join(flat_parts)


# Regex to parse HTTP Range header
RANGE_REGEX = re.compile(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", re.IGNORECASE)


def _adjust_range_header(original_range: str | None, resume_size: int) -> str | None:
    """
    Adjust HTTP Range header to account for resume position.
    """
    if not original_range:
        return f"bytes={resume_size}-"

    if "," in original_range:
        raise ValueError(f"Multiple ranges detected - {original_range!r}, not supported yet.")

    match = RANGE_REGEX.fullmatch(original_range)
    if not match:
        raise RuntimeError(f"Invalid range format - {original_range!r}.")
    start, end = match.groups()

    if not start:
        if not end:
            raise RuntimeError(f"Invalid range format - {original_range!r}.")

        new_suffix = int(end) - resume_size
        new_range = f"bytes=-{new_suffix}"
        if new_suffix <= 0:
            raise RuntimeError(f"Empty new range - {new_range!r}.")
        return new_range

    start = int(start)
    new_start = start + resume_size
    if end:
        end = int(end)
        new_range = f"bytes={new_start}-{end}"
        if new_start > end:
            raise RuntimeError(f"Empty new range - {new_range!r}.")
        return new_range

    return f"bytes={new_start}-"
<|endoftext|># Copyright 2019-present, the HuggingFace Inc. team.
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
"""Git LFS related utilities"""

import io
import os
from contextlib import AbstractContextManager
from typing import BinaryIO


class SliceFileObj(AbstractContextManager):
    """
    Utility context manager to read a *slice* of a seekable file-like object as a seekable, file-like object.

    This is NOT thread safe

    Inspired by stackoverflow.com/a/29838711/593036

    Credits to @julien-c

    Args:
        fileobj (`BinaryIO`):
            A file-like object to slice. MUST implement `tell()` and `seek()` (and `read()` of course).
            `fileobj` will be reset to its original position when exiting the context manager.
        seek_from (`int`):
            The start of the slice (offset from position 0 in bytes).
        read_limit (`int`):
            The maximum number of bytes to read from the slice.

    Attributes:
        previous_position (`int`):
            The previous position

    Examples:

    Reading 200 bytes with an offset of 128 bytes from a file (ie bytes 128 to 327):
    ```python
    >>> with open("path/to/file", "rb") as file:
    ...     with SliceFileObj(file, seek_from=128, read_limit=200) as fslice:
    ...         fslice.read(...)
    ```

    Reading a file in chunks of 512 bytes
    ```python
    >>> import os
    >>> chunk_size = 512
    >>> file_size = os.getsize("path/to/file")
    >>> with open("path/to/file", "rb") as file:
    ...     for chunk_idx in range(ceil(file_size / chunk_size)):
    ...         with SliceFileObj(file, seek_from=chunk_idx * chunk_size, read_limit=chunk_size) as fslice:
    ...             chunk = fslice.read(...)

    ```
    """

    def __init__(self, fileobj: BinaryIO, seek_from: int, read_limit: int):
        self.fileobj = fileobj
        self.seek_from = seek_from
        self.read_limit = read_limit

    def __enter__(self):
        self._previous_position = self.fileobj.tell()
        end_of_stream = self.fileobj.seek(0, os.SEEK_END)
        self._len = min(self.read_limit, end_of_stream - self.seek_from)
        # ^^ The actual number of bytes that can be read from the slice
        self.fileobj.seek(self.seek_from, io.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.seek(self._previous_position, io.SEEK_SET)

    def read(self, n: int = -1):
        pos = self.tell()
        if pos >= self._len:
            return b""
        remaining_amount = self._len - pos
        data = self.fileobj.read(remaining_amount if n < 0 else min(n, remaining_amount))
        return data

    def tell(self) -> int:
        return self.fileobj.tell() - self.seek_from

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        start = self.seek_from
        end = start + self._len
        if whence in (os.SEEK_SET, os.SEEK_END):
            offset = start + offset if whence == os.SEEK_SET else end + offset
            offset = max(start, min(offset, end))
            whence = os.SEEK_SET
        elif whence == os.SEEK_CUR:
            cur_pos = self.fileobj.tell()
            offset = max(start - cur_pos, min(offset, end - cur_pos))
        else:
            raise ValueError(f"whence value {whence} is not supported")
        return self.fileobj.seek(offset, whence) - self.seek_from

    def __iter__(self):
        while data := self.read(n=4 * 1024 * 1024):
            yield data
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
"""Contains utilities to handle datetimes in Huggingface Hub."""

from datetime import datetime, timezone


def parse_datetime(date_string: str) -> datetime:
    """
    Parses a date_string returned from the server to a datetime object.

    This parser is a weak-parser is the sense that it handles only a single format of
    date_string. It is expected that the server format will never change. The
    implementation depends only on the standard lib to avoid an external dependency
    (python-dateutil). See full discussion about this decision on PR:
    https://github.com/huggingface/huggingface_hub/pull/999.

    Example:
        ```py
        > parse_datetime('2022-08-19T07:19:38.123Z')
        datetime.datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=timezone.utc)
        ```

    Args:
        date_string (`str`):
            A string representing a datetime returned by the Hub server.
            String is expected to follow '%Y-%m-%dT%H:%M:%S.%fZ' pattern.

    Returns:
        A python datetime object.

    Raises:
        :class:`ValueError`:
            If `date_string` cannot be parsed.
    """
    try:
        # Normalize the string to always have 6 digits of fractional seconds
        if date_string.endswith("Z"):
            # Case 1: No decimal point (e.g., "2024-11-16T00:27:02Z")
            if "." not in date_string:
                # No fractional seconds - insert .000000
                date_string = date_string[:-1] + ".000000Z"
            # Case 2: Has decimal point (e.g., "2022-08-19T07:19:38.123456789Z")
            else:
                # Get the fractional and base parts
                base, fraction = date_string[:-1].split(".")
                # fraction[:6] takes first 6 digits and :0<6 pads with zeros if less than 6 digits
                date_string = f"{base}.{fraction[:6]:0<6}Z"

        return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(
            f"Cannot parse '{date_string}' as a datetime. Date string is expected to"
            " follow '%Y-%m-%dT%H:%M:%S.%fZ' pattern."
        ) from e
<|endoftext|># Copyright 2026-present, the HuggingFace Inc. team.
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
"""Centralized parser for Hugging Face Hub URIs ('hf://...') and mount specifications.

A HF URI is a URI-like string that identifies a location on the Hugging Face
Hub: a model/dataset/space/kernel repository, a bucket, optionally a revision,
and optionally a path inside the repo or bucket.

Canonical syntax:

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```

For convenience, [`parse_hf_uri`] also accepts Hugging Face **web URLs** (the
ones you copy-paste from your browser), e.g.
'https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv'. They are
normalized to the canonical 'hf://' form before parsing. Only unambiguous URLs
(repository / bucket pages and file/folder viewer routes) are accepted; any other
route is rejected rather than guessed.

A HF mount wraps a HF URI with a local mount path and an optional ':ro'/':rw'
flag (used by Spaces and Jobs volumes):

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

See 'docs/source/en/package_reference/hf_uris.md' for the full grammar and examples.
"""

import functools
import re
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

from huggingface_hub import constants
from huggingface_hub.errors import HfUriError, HFValidationError

from ._validators import validate_repo_id


# Inverse map (singular -> plural URI prefix). Built once from the canonical
# 'constants.HF_URI_TYPE_PREFIXES' and used to render URIs.
_TYPE_TO_PREFIX: dict[str, str] = {v: k for k, v in constants.HF_URI_TYPE_PREFIXES.items()}

# Special revisions that contain a '/'. They take precedence when splitting
# the part after '@' into '<revision>/<path-in-repo>'. Matches 'refs/pr/N'
# (Pull Request refs) and 'refs/convert/<name>' (e.g. parquet conversions).
# The conversion name allows the typical git ref characters '[a-zA-Z0-9_.-]'
# so names like 'parquet-v2' or 'duckdb.v1' round-trip correctly.
_SPECIAL_REFS_REVISION_REGEX = re.compile(r"^refs/(?:convert/[\w.-]+|pr/\d+)")

# Same as constants.HfUriType, but as a set of strings for easy lookup.)
_VALID_URI_TYPES: frozenset[str] = frozenset(constants.HF_URI_TYPE_PREFIXES.values())


# Web-viewer routes that point at a file or folder and that map cleanly onto a
# '<revision>/<path>' pair. Other routes (commit, commits, discussions, settings,
# edit, ...) do not identify a Hub location and are rejected by the URL parser.
_URL_REPO_LOCATION_ACTIONS: frozenset[str] = frozenset({"blame", "blob", "raw", "resolve", "tree"})
# Bucket web routes that point at a file or folder. Buckets are not versioned, so
# these are followed directly by '<path>' (no revision segment).
_URL_BUCKET_LOCATION_ACTIONS: frozenset[str] = frozenset({"resolve", "tree"})


@dataclass(frozen=True)
class HfUri:
    """Parsed representation of a Hugging Face Hub URI ('hf://...').

    Attributes:
        type (`str`):
            One of 'model', 'dataset', 'space', 'kernel' or 'bucket'.
        id (`str`):
            The repository id ('namespace/name', e.g. 'my-org/my-model') for repo URIs, or the bucket id ('namespace/name') for bucket URIs.
        revision (`str`, *optional*):
            The revision specified after '@' in the URI, URL-decoded. 'None' if no revision was specified, or for bucket URIs (which
            never carry a revision). Special refs like 'refs/pr/10' and 'refs/convert/parquet' are preserved as-is.
        path_in_repo (`str`):
            The path inside the repo or bucket. Empty string if the URI points at the root.
    """

    type: constants.HfUriType
    id: str
    revision: str | None = None
    path_in_repo: str = ""
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        uri = self._raw or ""  # For error messages

        # Check valid URI type
        if self.type not in _VALID_URI_TYPES:
            raise HfUriError(uri=uri, msg=f"Invalid type '{self.type}'. Must be one of {sorted(_VALID_URI_TYPES)}.")

        # Check valid ID
        if not self.id or self.id.count("/") != 1:
            raise HfUriError(uri=uri, msg=f"Id must be 'namespace/name', got '{self.id}'.")
        if self.type != "bucket":
            try:
                validate_repo_id(self.id)
            except HFValidationError as e:
                raise HfUriError(uri=uri, msg=str(e)) from e

        # Check valid revision
        if self.revision is not None and not self.revision:
            raise HfUriError(uri=uri, msg="Revision must not be an empty string.")
        if self.type == "bucket" and self.revision is not None:
            raise HfUriError(uri=uri, msg="Bucket URIs do not support a revision.")

        # Check valid path in repo
        if self.path_in_repo:
            if self.path_in_repo.startswith("/") or "//" in self.path_in_repo:
                raise HfUriError(uri=uri, msg=f"Path must not contain empty segments (got '{self.path_in_repo}').")

    @property
    def is_bucket(self) -> bool:
        """True if this URI points at a bucket."""
        return self.type == "bucket"

    @property
    def is_repo(self) -> bool:
        """True if this URI points at a repository (model, dataset, space or kernel)."""
        return self.type != "bucket"

    def to_uri(self) -> str:
        """Render the URI as a canonical 'hf://' string.

        The type prefix is always written explicitly (e.g. 'hf://models/my-org/my-model').
        """
        parts: list[str] = [constants.HF_PROTOCOL, _TYPE_TO_PREFIX[self.type], "/", self.id]
        if self.revision is not None:
            # Encode '/' as '%2F' for revisions that would otherwise be split as '<revision>/<path>'
            # at parse time. Special refs ('refs/pr/N', 'refs/convert/<name>') are kept verbatim
            # because the parser matches them eagerly.
            revision = self.revision
            if "/" in revision and _SPECIAL_REFS_REVISION_REGEX.fullmatch(revision) is None:
                revision = revision.replace("/", "%2F")
            parts.append(f"@{revision}")
        if self.path_in_repo:
            parts.append(f"/{self.path_in_repo}")
        return "".join(parts)

    def to_url(self, endpoint: str | None = None) -> str:
        """Render the URI as a Hugging Face **web URL** (the kind you open in a browser).

        This is the inverse of parsing a URL with [`parse_hf_uri`]. The returned URL points at:

        - the repository / bucket landing page when no path or revision is set;
        - the folder viewer ('/tree/<revision>') when only a revision is set;
        - the file viewer ('/blob/<revision>/<path>') for repository files (revision defaults to 'main');
        - the tree route ('/tree/<path>') for bucket files (buckets are not versioned).

        Args:
            endpoint (`str`, *optional*):
                Base endpoint to use. Defaults to 'constants.ENDPOINT' (i.e. 'https://huggingface.co').

        Returns:
            `str`: the web URL.

        Example:
            ```py
            >>> from huggingface_hub import parse_hf_uri
            >>> parse_hf_uri("hf://datasets/my-org/my-dataset@v1/train.csv").to_url()
            'https://huggingface.co/datasets/my-org/my-dataset/blob/v1/train.csv'
            ```
        """
        base = (endpoint or constants.ENDPOINT).rstrip("/")
        # Percent-encode characters that would otherwise break the URL (spaces, '#', '?', ...),
        # keeping '/' as the path separator. This is the inverse of the decoding done when parsing.
        path = quote(self.path_in_repo, safe="/")

        if self.type == "bucket":
            url = f"{base}/buckets/{self.id}"
            if path:
                url += f"/tree/{path}"
            return url

        # Models live at the root ('hf.co/<id>'); other repos are namespaced by their plural prefix.
        url = f"{base}/{self.id}" if self.type == "model" else f"{base}/{_TYPE_TO_PREFIX[self.type]}/{self.id}"
        revision = self.revision
        # Percent-encode the branch/tag name so it stays a single, URL-safe segment: a '/' would
        # otherwise open a new path segment and '#'/'?' would be read as a fragment/query when the
        # URL is opened or parsed back. This mirrors the decoding done when parsing. Special refs
        # ('refs/pr/N', 'refs/convert/<name>') are used verbatim by the Hub web routes.
        if revision is not None and _SPECIAL_REFS_REVISION_REGEX.fullmatch(revision) is None:
            revision = quote(revision, safe="")
        if path:
            url += f"/blob/{revision or constants.DEFAULT_REVISION}/{path}"
        elif revision is not None:
            url += f"/tree/{revision}"
        return url


@dataclass(frozen=True)
class HfMount:
    """A HF URI paired with a local mount path and optional read-only flag.

    Used by Spaces and Jobs to describe volume mounts. The full syntax is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    Attributes:
        source ([`HfUri`]):
            The parsed HF URI identifying the Hub resource to mount.
        mount_path (`str`):
            The local mount path (always starts with '/').
        read_only (`bool`, *optional*):
            True if the mount ends with ':ro', False if it ends with ':rw', 'None' if no flag was provided.
    """

    source: HfUri
    mount_path: str
    read_only: bool | None = None
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        raw = self._raw or ""
        if not self.mount_path.startswith("/") or self.mount_path == "/":
            raise HfUriError(
                uri=raw,
                msg=f"Mount path must be a non-empty absolute path starting with '/', got '{self.mount_path}'.",
            )

    def to_uri(self) -> str:
        """Render the mount as a canonical 'hf://' string.

        Example: 'hf://models/my-org/my-model:/data:ro'
        """
        parts = [self.source.to_uri(), ":", self.mount_path]
        if self.read_only is not None:
            parts.append(":ro" if self.read_only else ":rw")
        return "".join(parts)


def is_hf_uri(uri: str) -> bool:
    """Check if a string is a valid HF URI ('hf://...') or a recognized Hugging Face web URL."""
    try:
        parse_hf_uri(uri)
        return True
    except HfUriError:
        return False


@functools.lru_cache
def parse_hf_uri(uri: str, endpoint: str | None = None) -> HfUri:
    """Parse a Hugging Face Hub URI ('hf://...') or a Hugging Face web URL.

    A HF URI is a URI-like string identifying a location on the Hugging Face Hub. The full grammar is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
    ```

    For convenience, Hugging Face **web URLs** (the ones you copy-paste from the website) are also
    accepted and normalized to the canonical 'hf://' form, e.g.
    'https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv'. Only unambiguous URLs
    (repository / bucket pages and file/folder viewer routes) are accepted; any other route is rejected.

    See 'docs/source/en/package_reference/hf_uris.md' for the full specification.

    Args:
        uri (`str`):
            The URI to parse. Must start with 'hf://', or be a Hugging Face URL (e.g. 'https://huggingface.co/...').
        endpoint (`str`, *optional*):
            A custom Hub endpoint (e.g. a self-hosted or proxied Hub like 'https://hub.my-company.com' or
            'http://localhost:8080/hf'). When provided, web URLs on that endpoint are recognized in addition to
            the default Hugging Face hosts. Has no effect on 'hf://' URIs.

    Returns:
        [`HfUri`]: the parsed URI.

    Raises:
        [`HfUriError`]:
            If the URI is malformed (missing prefix, invalid type, missing id, unsupported URL route, etc.).

    Examples:
        ```py
        >>> from huggingface_hub.utils import parse_hf_uri
        >>> parse_hf_uri("hf://my-org/my-model")
        HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo='')
        >>> parse_hf_uri("hf://datasets/my-org/my-dataset@refs/pr/3/train.json")
        HfUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
        >>> parse_hf_uri("https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv")
        HfUri(type='dataset', id='my-org/my-dataset', revision='main', path_in_repo='train.csv')
        ```
    """
    raw = uri
    if uri.startswith(constants.HF_PROTOCOL):
        body = uri[len(constants.HF_PROTOCOL) :]
        if not body:
            raise HfUriError(uri, f"Empty body after '{constants.HF_PROTOCOL}'.")
    elif _looks_like_hf_url(uri, endpoint=endpoint):
        body = _url_to_uri_body(uri, endpoint=endpoint)
    else:
        raise HfUriError(
            uri,
            f"Must start with '{constants.HF_PROTOCOL}' or be a Hugging Face URL (e.g. 'https://huggingface.co/...'). "
            f"Expected format: {constants.HF_PROTOCOL}[<TYPE>/]<ID>[@<REVISION>][/<PATH>]",
        )

    type_, location = _split_type(body, raw=raw)

    if type_ == "bucket":
        return _parse_bucket_body(location, type_, raw=raw)
    return _parse_repo_body(location, type_, raw=raw)


def _endpoint_host_and_path(endpoint: str | None) -> tuple[str | None, str]:
    """Return the lowercased host and stripped path prefix of a custom Hub 'endpoint'.

    E.g. 'https://hub.my-company.com' -> ('hub.my-company.com', '') and a self-hosted
    'http://localhost:8080/hf' -> ('localhost', 'hf'). Returns '(None, "")' when 'endpoint' is None.
    """
    if endpoint is None:
        return None, ""
    # Prefix '//' for scheme-less endpoints so 'urlsplit' populates 'netloc' instead of 'path'.
    parsed = urlsplit(endpoint if "://" in endpoint else "//" + endpoint)
    host = parsed.hostname.lower() if parsed.hostname else None
    return host, parsed.path.strip("/")


def _recognized_hosts(endpoint: str | None) -> frozenset[str]:
    """The set of hosts whose web URLs can be parsed: the default Hugging Face hosts plus 'endpoint'."""
    host, _ = _endpoint_host_and_path(endpoint)
    return constants.HF_URL_HOSTS | {host} if host else constants.HF_URL_HOSTS


def _looks_like_hf_url(uri: str, endpoint: str | None = None) -> bool:
    """Return True if 'uri' looks like a (possibly scheme-less) Hugging Face web URL."""
    lowered = uri.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    # Scheme-less host (e.g. 'huggingface.co/org/model').
    return any(lowered == host or lowered.startswith(host + "/") for host in _recognized_hosts(endpoint))


def _decode_url_path_segment(segment: str) -> str:
    """Percent-decode a single URL path segment (e.g. 'file%20name.txt' -> 'file name.txt').

    A decoded '/' is re-encoded as '%2F' so the segment stays atomic when the normalized body is
    re-split by the shared parser. This decodes ordinary path characters (spaces, '#', ...) that
    browsers encode, while keeping '%2F'-encoded revisions (e.g. 'feature%2Ffoo') intact.
    """
    return unquote(segment).replace("/", "%2F")


def _url_to_uri_body(url: str, endpoint: str | None = None) -> str:
    """Normalize a Hugging Face web URL into the body of a 'hf://' URI (everything after 'hf://').

    The returned string is fed back into the regular URI parsing logic, so all validation
    (repo id, revision, empty path segments, ...) is shared with the canonical 'hf://' path.
    Only unambiguous URLs are accepted: any unrecognized route raises [`HfUriError`]. When 'endpoint'
    is provided, URLs on that custom Hub host are recognized too (and its path prefix is stripped).
    """
    raw = url
    # Prefix '//' for scheme-less inputs so 'urlsplit' populates 'netloc' instead of 'path'.
    parsed = urlsplit(url if "://" in url else "//" + url)
    host = (parsed.hostname or "").lower()
    if host not in _recognized_hosts(endpoint):
        raise HfUriError(
            uri=raw,
            msg=f"Unrecognized host '{host or url}'. Expected a Hugging Face URL (e.g. 'https://huggingface.co/...').",
        )

    # Query string and fragment are intentionally dropped (e.g. '?download=true').
    path = parsed.path
    # For a self-hosted endpoint with a path prefix (e.g. 'http://localhost:8080/hf'), drop it so the
    # remaining segments are '[<TYPE>/]<namespace>/<name>[/...]' just like on the public Hub.
    endpoint_host, endpoint_path = _endpoint_host_and_path(endpoint)
    if endpoint_path and host == endpoint_host:
        prefix = "/" + endpoint_path
        if path == prefix or path.startswith(prefix + "/"):
            path = path[len(prefix) :]
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        raise HfUriError(uri=raw, msg=f"Missing repository or bucket identifier in URL '{url}'.")

    # Optional type prefix ('datasets', 'spaces', 'kernels', 'buckets', 'models').
    type_prefix: str | None = None
    if segments[0] in constants.HF_URI_TYPE_PREFIXES:
        type_prefix = segments[0]
        segments = segments[1:]

    # Everything in the web UI is namespaced ('<namespace>/<name>'); a single segment is a user or
    # organization page (or a listing page), which we cannot map to a repository -> reject.
    if len(segments) < 2:
        raise HfUriError(
            uri=raw,
            msg=(
                f"Cannot parse URL '{url}': expected a '<namespace>/<name>' repository or bucket. "
                "User/organization pages and single-segment URLs are not supported."
            ),
        )
    repo_id = f"{segments[0]}/{segments[1]}"
    rest = segments[2:]

    if type_prefix == "buckets":
        if not rest:
            return f"buckets/{repo_id}"
        action, *tail = rest
        if action not in _URL_BUCKET_LOCATION_ACTIONS:
            raise HfUriError(uri=raw, msg=f"Cannot parse bucket URL '{url}': unsupported '/{action}/' route.")
        path = "/".join(_decode_url_path_segment(segment) for segment in tail)
        return f"buckets/{repo_id}/{path}" if path else f"buckets/{repo_id}"

    prefix = f"{type_prefix}/" if type_prefix else ""
    if not rest:
        return f"{prefix}{repo_id}"
    action, *tail = rest
    if action not in _URL_REPO_LOCATION_ACTIONS:
        raise HfUriError(
            uri=raw,
            msg=(
                f"Cannot parse URL '{url}': unsupported '/{action}/' route. "
                "Only repository pages and file/folder viewer routes (blob, resolve, raw, tree, ...) can be parsed."
            ),
        )
    if not tail:
        # e.g. '.../tree' with nothing after -> repository root.
        return f"{prefix}{repo_id}"
    # 'tail' is '<revision>/<path>'; reuse the canonical '@<revision>/<path>' splitting logic
    # (special refs, URL-encoded slashes, ...) by handing it back to the URI parser. Each segment
    # is percent-decoded first so file names with spaces, '#', ... resolve correctly; the revision
    # segment's '%2F' survives (re-encoded by '_decode_url_path_segment') and is decoded downstream.
    decoded = "/".join(_decode_url_path_segment(segment) for segment in tail)
    return f"{prefix}{repo_id}@{decoded}"


def parse_hf_mount(mount_str: str) -> HfMount:
    """Parse a HF mount specification ('hf://...:<MOUNT_PATH>[:ro|:rw]').

    A mount specification is a HF URI followed by a local mount path and an optional read-only/read-write flag.
    The full grammar is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    See 'docs/source/en/package_reference/hf_uris.md' for the full specification.

    Args:
        mount_str (`str`):
            The mount string to parse. Must start with 'hf://' and contain a ':<MOUNT_PATH>' segment.

    Returns:
        [`HfMount`]: the parsed mount.

    Raises:
        [`HfUriError`]:
            If the mount string is malformed (missing mount path, invalid URI, etc.).

    Examples:
        ```py
        >>> from huggingface_hub.utils import parse_hf_mount
        >>> parse_hf_mount("hf://my-org/my-model:/data:ro")
        HfMount(source=HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo=''), mount_path='/data', read_only=True)
        >>> parse_hf_mount("hf://buckets/my-org/my-bucket/sub/dir:/mnt:rw")
        HfMount(source=HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=False)
        ```
    """
    if not mount_str.startswith(constants.HF_PROTOCOL):
        raise HfUriError(
            uri=mount_str,
            msg=f"Must start with '{constants.HF_PROTOCOL}'.",
        )

    raw = mount_str
    body = mount_str[len(constants.HF_PROTOCOL) :]
    if not body:
        raise HfUriError(uri=raw, msg=f"Empty body after '{constants.HF_PROTOCOL}'.")

    location, mount_path, read_only = _split_mount(body, raw=raw)

    if mount_path is None:
        raise HfUriError(uri=raw, msg="Missing mount path. Expected ':<MOUNT_PATH>' (e.g. 'hf://org/model:/data').")

    # Re-assemble the URI part and parse it
    uri_str = constants.HF_PROTOCOL + location
    try:
        source = parse_hf_uri(uri_str)
    except HfUriError as e:
        raise HfUriError(uri=raw, msg=e.msg) from e

    return HfMount(source=source, mount_path=mount_path, read_only=read_only, _raw=raw)


def _split_mount(body: str, *, raw: str) -> tuple[str, str | None, bool | None]:
    """Split the ':<MOUNT_PATH>[:ro|:rw]' suffix from 'body'.

    Returns '(location, mount_path, read_only)' where 'mount_path' is 'None' if no mount segment is present.
    """
    if body.endswith(":ro"):
        read_only, body = True, body.removesuffix(":ro")
    elif body.endswith(":rw"):
        read_only, body = False, body.removesuffix(":rw")
    else:
        read_only = None

    # Mount paths always start with '/', so the delimiter is ':/'.
    # We use rfind() because the mount segment is always trailing
    idx = body.rfind(":/")
    if idx == -1:
        if read_only is not None:
            raise HfUriError(
                uri=raw,
                msg="':ro'/':rw' suffix is only valid when a mount path is provided (e.g. 'hf://...:/<MOUNT_PATH>:ro').",
            )
        return body, None, None

    location = body[:idx]
    mount_path = body[idx + 1 :]  # includes the leading '/'
    if not location:
        raise HfUriError(uri=raw, msg="Missing location before mount path.")
    return location, mount_path, read_only


def _split_type(location: str, *, raw: str) -> tuple[constants.HfUriType, str]:
    """Detect the (optional) type prefix and return '(type, remaining_location)'.

    A missing type prefix defaults to 'model'. Singular forms ('model/', 'dataset/', etc.) are explicitly rejected with a helpful error.
    """
    slash_idx = location.find("/")
    if slash_idx == -1:
        # Single segment, no prefix. Reject if it looks like a bare type name.
        if location in constants.HF_URI_TYPE_PREFIXES:
            raise HfUriError(
                uri=raw,
                msg=f"Missing identifier after '{location}'. Expected '{constants.HF_PROTOCOL}{location}/<ID>'.",
            )
        if (singular_plural := _TYPE_TO_PREFIX.get(location)) is not None:
            raise HfUriError(
                uri=raw,
                msg=f"Type prefix must be plural. Did you mean '{constants.HF_PROTOCOL}{singular_plural}/...'?",
            )
        return "model", location

    first = location[:slash_idx]
    rest = location[slash_idx + 1 :]
    if first in constants.HF_URI_TYPE_PREFIXES:
        return constants.HF_URI_TYPE_PREFIXES[first], rest
    if (singular_plural := _TYPE_TO_PREFIX.get(first)) is not None:
        raise HfUriError(
            uri=raw, msg=f"Type prefix must be plural, got '{first}/'. Did you mean '{singular_plural}/'?"
        )
    return "model", location


def _parse_bucket_body(
    location: str,
    type_: constants.HfUriType,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a bucket URI: 'namespace/name[/path]'."""
    location = location.strip("/")
    parts = location.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise HfUriError(uri=raw, msg=f"Bucket id must be 'namespace/name', got '{location}'.")
    bucket_id = f"{parts[0]}/{parts[1]}"
    if "@" in bucket_id:
        raise HfUriError(uri=raw, msg="Bucket URIs do not support a revision marker ('@').")
    path_in_bucket = parts[2] if len(parts) >= 3 else ""
    return HfUri(
        type=type_,
        id=bucket_id,
        revision=None,
        path_in_repo=path_in_bucket,
        _raw=raw,
    )


def _parse_repo_body(
    location: str,
    type_: constants.HfUriType,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a repo URI: '<repo_id>[@<revision>][/<path>]'."""
    location = location.strip("/")
    if not location:
        raise HfUriError(uri=raw, msg="Missing repository id.")

    # The '@' separates the repo_id from the revision, but only when it
    # appears right after 'namespace/name' (at most one '/' before it).
    # An '@' deeper in the path (e.g. in a filename like 'file@1.txt') is literal.
    at_idx = location.find("@")
    revision: str | None

    if at_idx == -1 or location[:at_idx].count("/") > 1:
        # No '@' at all, or the '@' is past the repo_id portion (in a filename).
        revision = None
        parts = location.split("/", 2)
        if len(parts) < 2:
            raise HfUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{location}'. ")
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = parts[2] if len(parts) > 2 else ""
    else:
        repo_id = location[:at_idx]
        rev_and_path = location[at_idx + 1 :]
        if not repo_id:
            raise HfUriError(uri=raw, msg="Missing repository id before '@'.")
        if repo_id.count("/") != 1:
            raise HfUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{repo_id}'.")
        # Special refs like 'refs/pr/10' contain '/' and must be matched eagerly,
        # otherwise we would split them at the first '/' and treat the rest as a path.
        match = _SPECIAL_REFS_REVISION_REGEX.search(rev_and_path)
        if match is not None:
            revision = match.group()
            path_in_repo = rev_and_path[len(revision) :].removeprefix("/")
        else:
            slash_idx = rev_and_path.find("/")
            if slash_idx == -1:
                revision = rev_and_path
                path_in_repo = ""
            else:
                revision = rev_and_path[:slash_idx]
                path_in_repo = rev_and_path[slash_idx + 1 :]
        revision = unquote(revision)
        if not revision:
            raise HfUriError(uri=raw, msg="Empty revision after '@'.")

    return HfUri(
        type=type_,
        id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        _raw=raw,
    )
<|endoftext|>from queue import Queue
from threading import Lock, Thread
from urllib.parse import quote

from .. import constants, logging
from . import build_hf_headers, get_session, hf_raise_for_status


logger = logging.get_logger(__name__)

# Telemetry is sent by a separate thread to avoid blocking the main thread.
# A daemon thread is started once and consume tasks from the _TELEMETRY_QUEUE.
# If the thread stops for some reason -shouldn't happen-, we restart a new one.
_TELEMETRY_THREAD: Thread | None = None
_TELEMETRY_THREAD_LOCK = Lock()  # Lock to avoid starting multiple threads in parallel
_TELEMETRY_QUEUE: Queue = Queue()


def send_telemetry(
    topic: str,
    *,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
) -> None:
    """
    Sends telemetry that helps track usage of different HF libraries.

    This usage data helps us debug issues and prioritize new features. However, we understand that not everyone wants
    to share additional information, and we respect your privacy. You can disable telemetry collection by setting the
    `HF_HUB_DISABLE_TELEMETRY=1` as environment variable. Telemetry is also disabled in offline mode (i.e. when setting
    `HF_HUB_OFFLINE=1`).

    Telemetry collection is run in a separate thread to minimize impact for the user.

    Args:
        topic (`str`):
            Name of the topic that is monitored. The topic is directly used to build the URL. If you want to monitor
            subtopics, just use "/" separation. Examples: "gradio", "transformers/examples",...
        library_name (`str`, *optional*):
            The name of the library that is making the HTTP request. Will be added to the user-agent header.
        library_version (`str`, *optional*):
            The version of the library that is making the HTTP request. Will be added to the user-agent header.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string. It will be completed with information about the installed packages.

    Example:
    ```py
    >>> from huggingface_hub.utils import send_telemetry

    # Send telemetry without library information
    >>> send_telemetry("ping")

    # Send telemetry to subtopic with library information
    >>> send_telemetry("gradio/local_link", library_name="gradio", library_version="3.22.1")

    # Send telemetry with additional data
    >>> send_telemetry(
    ...     topic="examples",
    ...     library_name="transformers",
    ...     library_version="4.26.0",
    ...     user_agent={"pipeline": "text_classification", "framework": "flax"},
    ... )
    ```
    """
    if constants.is_offline_mode() or constants.HF_HUB_DISABLE_TELEMETRY:
        return

    _start_telemetry_thread()  # starts thread only if doesn't exist yet
    _TELEMETRY_QUEUE.put(
        {"topic": topic, "library_name": library_name, "library_version": library_version, "user_agent": user_agent}
    )


def _start_telemetry_thread():
    """Start a daemon thread to consume tasks from the telemetry queue.

    If the thread is interrupted, start a new one.
    """
    with _TELEMETRY_THREAD_LOCK:  # avoid to start multiple threads if called concurrently
        global _TELEMETRY_THREAD
        if _TELEMETRY_THREAD is None or not _TELEMETRY_THREAD.is_alive():
            _TELEMETRY_THREAD = Thread(target=_telemetry_worker, daemon=True)
            _TELEMETRY_THREAD.start()


def _telemetry_worker():
    """Wait for a task and consume it."""
    while True:
        kwargs = _TELEMETRY_QUEUE.get()
        _send_telemetry_in_thread(**kwargs)
        _TELEMETRY_QUEUE.task_done()


def _send_telemetry_in_thread(
    topic: str,
    *,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
) -> None:
    """Contains the actual data sending data to the Hub.

    This function is called directly in gradio's analytics because
    it is not possible to send telemetry from a daemon thread.

    See here: https://github.com/gradio-app/gradio/pull/8180

    Please do not rename or remove this function.
    """
    path = "/".join(quote(part) for part in topic.split("/") if len(part) > 0)
    try:
        r = get_session().head(
            f"{constants.ENDPOINT}/api/telemetry/{path}",
            headers=build_hf_headers(
                token=False,  # no need to send a token for telemetry
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
            ),
        )
        hf_raise_for_status(r)
    except Exception as e:
        # We don't want to error in case of connection errors of any kind.
        logger.debug(f"Error while sending telemetry: {e}")
<|endoftext|>import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from .. import constants
from ..file_download import repo_folder_name
from .sha import git_hash, sha_fileobj


if TYPE_CHECKING:
    from ..hf_api import RepoFile

# using fullmatch for clarity and strictness
_REGEX_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


# Typed structure describing a checksum mismatch
class Mismatch(TypedDict):
    path: str
    expected: str
    actual: str
    algorithm: str


HashAlgo = Literal["sha256", "git-sha1"]


@dataclass(frozen=True)
class FolderVerification:
    revision: str
    checked_count: int
    mismatches: list[Mismatch]
    missing_paths: list[str]
    extra_paths: list[str]
    verified_path: Path


def collect_local_files(root: Path) -> dict[str, Path]:
    """
    Return a mapping of repo-relative path -> absolute path for all files under `root`.
    """
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


def _resolve_commit_hash_from_cache(storage_folder: Path, revision: str | None) -> str:
    """
    Resolve a commit hash from a cache repo folder and an optional revision.
    """
    if revision and _REGEX_COMMIT_HASH.fullmatch(revision):
        return revision

    refs_dir = storage_folder / "refs"
    snapshots_dir = storage_folder / "snapshots"

    if revision:
        ref_path = refs_dir / revision
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        raise ValueError(f"Revision '{revision}' could not be resolved in cache (expected file '{ref_path}').")

    # No revision provided: try common defaults
    main_ref = refs_dir / "main"
    if main_ref.is_file():
        return main_ref.read_text(encoding="utf-8").strip()

    if not snapshots_dir.is_dir():
        raise ValueError(f"Cache repo is missing snapshots directory: {snapshots_dir}. Provide --revision explicitly.")

    candidates = [p.name for p in snapshots_dir.iterdir() if p.is_dir() and _REGEX_COMMIT_HASH.fullmatch(p.name)]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Ambiguous cached revision: multiple snapshots found and no refs to disambiguate. Please pass --revision."
    )


def compute_file_hash(path: Path, algorithm: HashAlgo) -> str:
    """
    Compute the checksum of a local file using the requested algorithm.
    """

    with path.open("rb") as stream:
        if algorithm == "sha256":
            return sha_fileobj(stream).hex()
        if algorithm == "git-sha1":
            return git_hash(stream.read())
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def verify_maps(
    *,
    remote_by_path: dict[str, "RepoFile"],
    local_by_path: dict[str, Path],
    revision: str,
    verified_path: Path,
) -> FolderVerification:
    """Compare remote entries and local files and return a verification result."""
    remote_paths = set(remote_by_path)
    local_paths = set(local_by_path)

    missing = sorted(remote_paths - local_paths)
    extra = sorted(local_paths - remote_paths)
    both = sorted(remote_paths & local_paths)

    mismatches: list[Mismatch] = []

    for rel_path in both:
        remote_entry = remote_by_path[rel_path]
        local_path = local_by_path[rel_path]

        lfs = getattr(remote_entry, "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_sha is None and isinstance(lfs, dict):
            lfs_sha = lfs.get("sha256")
        if lfs_sha:
            algorithm: HashAlgo = "sha256"
            expected = str(lfs_sha).lower()
        else:
            blob_id = remote_entry.blob_id  # type: ignore
            algorithm = "git-sha1"
            expected = str(blob_id).lower()

        actual = compute_file_hash(local_path, algorithm)

        if actual != expected:
            mismatches.append(Mismatch(path=rel_path, expected=expected, actual=actual, algorithm=algorithm))

    return FolderVerification(
        revision=revision,
        checked_count=len(both),
        mismatches=mismatches,
        missing_paths=missing,
        extra_paths=extra,
        verified_path=verified_path,
    )


def resolve_local_root(
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    cache_dir: Path | None,
    local_dir: Path | None,
) -> tuple[Path, str]:
    """
    Resolve the root directory to scan locally and the remote revision to verify.
    """
    if local_dir is not None:
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Local directory does not exist or is not a directory: {root}")
        return root, (revision or constants.DEFAULT_REVISION)

    cache_root = Path(cache_dir or constants.HF_HUB_CACHE).expanduser().resolve()
    storage_folder = cache_root / repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    if not storage_folder.exists():
        raise ValueError(
            f"Repo is not present in cache: {storage_folder}. Use 'hf download' first or pass --local-dir."
        )
    commit = _resolve_commit_hash_from_cache(storage_folder, revision)
    snapshot_dir = storage_folder / "snapshots" / commit
    if not snapshot_dir.is_dir():
        raise ValueError(f"Snapshot directory does not exist for revision '{commit}': {snapshot_dir}.")
    return snapshot_dir, commit
<|endoftext|># Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains commands to print information about the environment and version."""

import shutil
import subprocess
import sys

import click

from huggingface_hub import __version__, constants

from ..utils import dump_environment_info, installation_method
from ._cli_utils import _fetch_latest_pypi_version, run_update
from ._output import out


def env() -> None:
    """Print information about the environment."""
    dump_environment_info()


def version() -> None:
    """Print information about the hf version."""
    out.result("hf version", version=__version__)


def update() -> None:
    """Update the `hf` CLI to the latest version."""
    from ._skills import DEFAULT_SKILL_ID, _installed_hf_cli_dirs

    out.text(f"Current version: {__version__}")
    out.text("Checking for updates to latest version...")
    latest_version = _fetch_latest_pypi_version("huggingface_hub")
    if latest_version is not None and __version__ == latest_version:
        out.text(f"hf is up to date ({__version__})")
        return

    # Windows refuses to replace the `hf.exe` launcher of a running process: pip would fail with a
    # "file in use" error (WinError 32) *after* having uninstalled the current version, leaving a dead
    # `hf.exe` and no `huggingface_hub` module behind. Print the command instead of running it, the same
    # way pip refuses to upgrade itself on Windows.
    if sys.platform == "win32" and installation_method() == "pip":
        command = subprocess.list2cmdline([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
        out.error(
            "On Windows, a pip-installed `hf` cannot update itself: pip is not allowed to replace `hf.exe` while it "
            f"is running. Run this command instead:\n    {command}"
        )
        raise click.exceptions.Exit(code=1)

    # The standalone installer installs the `hf-cli` skill by default. If it's not installed at this
    # point, the user opted out (or removed it): tell the installer to leave it alone instead of
    # silently undoing that choice.
    skill_installed = bool(
        _installed_hf_cli_dirs(constants.AGENTS_SKILLS_GLOBAL_PATH, constants.CLAUDE_SKILLS_GLOBAL_PATH)
    )

    returncode = run_update(exclude_skill=not skill_installed)
    if returncode != 0:
        raise click.exceptions.Exit(code=returncode)

    if not skill_installed:
        out.hint("Run `hf skills add -g` to teach your AI agents how to use the `hf` CLI.")
        return

    # Refresh the globally installed skill so agents see the new command surface. Runs in a
    # subprocess: the skill is generated from the CLI code, which has just been replaced on disk
    # while this process still runs the previous version.
    out.text(f"Updating the `{DEFAULT_SKILL_ID}` skill...")
    subprocess.call([*_hf_argv(), "skills", "update", DEFAULT_SKILL_ID, "-g"])


def _hf_argv() -> list[str]:
    """argv prefix to invoke the freshly updated `hf` CLI."""
    if installation_method() == "brew":
        # Homebrew installs the new version in a new prefix: the current interpreter is the old one.
        hf_bin = shutil.which("hf")
        if hf_bin is not None:
            return [hf_bin]
    return [sys.executable, "-m", "huggingface_hub.cli.hf"]
<|endoftext|>"""Internal helpers for Hugging Face marketplace skill installation and upgrades."""

import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from huggingface_hub import __version__, constants
from huggingface_hub._buckets import BucketFile
from huggingface_hub.errors import CLIError

from ..utils import disable_progress_bars, logging
from ._cli_utils import get_hf_api
from ._output import out


logger = logging.get_logger(__name__)

DEFAULT_SKILL_ID = "hf-cli"
DEFAULT_SKILLS_BUCKET_ID = "huggingface/skills"
MARKETPLACE_PATH = "marketplace.json"
# Empty marker file dropped into managed skill installs so `hf skills update` knows
# to touch them and leave user-placed skill dirs alone. Filename is historical (used
# to be a JSON manifest with a revision); we keep it for backward compat with installs
# made by previous versions.
MANAGED_MARKER_FILENAME = ".hf-skill-manifest.json"

SkillUpdateStatus = Literal["up_to_date", "unmanaged", "source_unreachable"]


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    repo_path: str
    description: str | None = None


@dataclass(frozen=True)
class SkillUpdateInfo:
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None


def add_skill(skill_name: str, destination_root: Path, force: bool = False) -> Path:
    """Resolve a marketplace skill by name and install it."""
    api = get_hf_api()
    with disable_progress_bars():
        marketplace_skills = _load_marketplace_skills(api)
        skill = _select_marketplace_skill(marketplace_skills, skill_name)
        if skill is None:
            raise CLIError(
                f"Skill '{skill_name}' not found in {DEFAULT_SKILLS_BUCKET_ID}. "
                "Try `hf skills add` to install `hf-cli` or use a known skill name."
            )
        return _install_marketplace_skill(api, skill, destination_root, force=force)


def install_generated_skill(content: str, destination_root: Path, force: bool = False) -> Path:
    """Install the `hf-cli` skill from locally generated SKILL.md content (no bucket download)."""

    def populate(install_dir: Path) -> None:
        install_dir.mkdir(parents=True, exist_ok=True)
        (install_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (install_dir / MANAGED_MARKER_FILENAME).touch()

    return _install_skill(DEFAULT_SKILL_ID, destination_root, populate=populate, force=force)


def update_skills(roots: list[Path], selector: str | None = None, *, hf_cli_content: str) -> list[SkillUpdateInfo]:
    """Re-sync managed skill installs (`hf-cli` is rewritten from `hf_cli_content`, the rest from the bucket)."""
    skill_dirs = _iter_unique_skill_dirs(roots)
    if selector is not None:
        selector_lower = selector.strip().lower()
        skill_dirs = [d for d in skill_dirs if d.name.lower() == selector_lower]
        if not skill_dirs:
            raise CLIError(f"No installed skill matches '{selector}'. Install it with `hf skills add {selector}`.")

    # `hf-cli` is regenerated locally, so only hit the marketplace when another managed skill needs it.
    needs_marketplace = any(d.name != DEFAULT_SKILL_ID and (d / MANAGED_MARKER_FILENAME).exists() for d in skill_dirs)
    api = None
    marketplace_skills: dict[str, MarketplaceSkill] = {}
    if needs_marketplace:
        api = get_hf_api()
        with disable_progress_bars():
            marketplace_skills = {skill.name.lower(): skill for skill in _load_marketplace_skills(api)}

    return [_apply_single_update(api, skill_dir, marketplace_skills, hf_cli_content) for skill_dir in skill_dirs]


def _load_marketplace_skills(api) -> list[MarketplaceSkill]:
    payload = _load_marketplace_payload(api)
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        raise CLIError("Invalid marketplace payload: expected a top-level 'plugins' list.")

    skills: list[MarketplaceSkill] = []
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        name = plugin.get("name")
        source = plugin.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        description = plugin.get("description")
        skills.append(
            MarketplaceSkill(
                name=name,
                repo_path=_normalize_repo_path(source),
                description=description if isinstance(description, str) else None,
            )
        )
    return skills


def _install_marketplace_skill(api, skill: MarketplaceSkill, destination_root: Path, force: bool = False) -> Path:
    """Install a marketplace skill into a local skills directory."""

    def populate(install_dir: Path) -> None:
        install_dir.mkdir(parents=True, exist_ok=True)
        bucket_files = _list_skill_files(api, skill)
        _download_skill_files(api, skill, bucket_files, install_dir)
        _validate_installed_skill_dir(install_dir)
        (install_dir / MANAGED_MARKER_FILENAME).touch()

    return _install_skill(skill.name, destination_root, populate=populate, force=force)


_VALID_SKILL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _install_skill(
    name: str,
    destination_root: Path,
    populate: Callable[[Path], None],
    force: bool = False,
) -> Path:
    """Install a skill into ``destination_root`` by calling ``populate(install_dir)`` to fill it.

    Used by both the marketplace install (populate = download from bucket) and the
    locally-generated install (populate = write content). When the install already
    exists and ``force`` is set, the new content is staged in a sibling tempdir and
    atomically swapped in, so the existing install stays intact if ``populate``
    fails halfway through.
    """
    # `name` may come from the remote marketplace payload and the install dir is removed on
    # reinstall: validate it as defense-in-depth against path traversal.
    if not _VALID_SKILL_NAME.fullmatch(name):
        raise CLIError(f"Invalid skill name '{name}'.")
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    install_dir = destination_root / name
    already_exists = install_dir.exists()

    if already_exists and not force:
        raise FileExistsError(f"Skill already exists: {install_dir}")

    if already_exists:
        with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{install_dir.name}.install-") as tmp_dir_str:
            staged_dir = Path(tmp_dir_str) / install_dir.name
            populate(staged_dir)
            _atomic_replace_directory(existing_dir=install_dir, staged_dir=staged_dir)
        return install_dir

    try:
        populate(install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


def _load_marketplace_payload(api) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "marketplace.json"
        api.download_bucket_files(
            DEFAULT_SKILLS_BUCKET_ID,
            [(MARKETPLACE_PATH, local_path)],
            raise_on_missing_files=True,
        )
        parsed = json.loads(local_path.read_text(encoding="utf-8"))

    if not isinstance(parsed, dict):
        raise CLIError("Invalid marketplace payload: expected a JSON object.")
    return parsed


def _select_marketplace_skill(skills: list[MarketplaceSkill], selector: str) -> MarketplaceSkill | None:
    selector_lower = selector.strip().lower()
    for skill in skills:
        if skill.name.lower() == selector_lower:
            return skill
    return None


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    if not normalized:
        raise CLIError("Invalid marketplace entry: empty source path.")
    return normalized


def _validate_installed_skill_dir(skill_dir: Path) -> None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise RuntimeError(f"Installed skill is missing SKILL.md: {skill_file}")


def _list_skill_files(api, skill: MarketplaceSkill) -> list[BucketFile]:
    """List all files under `skill.repo_path` in the marketplace bucket."""
    prefix = skill.repo_path.rstrip("/")
    files: list[BucketFile] = [
        item
        for item in api.list_bucket_tree(DEFAULT_SKILLS_BUCKET_ID, prefix=prefix, recursive=True)
        if isinstance(item, BucketFile)
    ]
    if not files:
        raise FileNotFoundError(f"Path '{prefix}' not found in bucket '{DEFAULT_SKILLS_BUCKET_ID}'.")
    return files


def _download_skill_files(api, skill: MarketplaceSkill, files: list[BucketFile], install_dir: Path) -> None:
    """Download bucket files into `install_dir`."""
    prefix = skill.repo_path.rstrip("/")
    prefix_with_slash = f"{prefix}/"

    # `list_bucket_tree(prefix=...)` matches as a raw string prefix, so e.g. asking for
    # "skills/gradio" can also return "skills/gradio-tools/...". Filter on the trailing
    # slash to keep only files actually inside the directory, then strip it so files land
    # directly under `install_dir` preserving any nested structure.
    download_specs: list[tuple[str | BucketFile, str | Path]] = []
    for bucket_file in files:
        if not bucket_file.path.startswith(prefix_with_slash):
            continue
        relative = bucket_file.path[len(prefix_with_slash) :]
        local_file = install_dir.joinpath(*PurePosixPath(relative).parts)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        download_specs.append((bucket_file, local_file))

    if not download_specs:
        raise FileNotFoundError(f"No files found under '{prefix}' in bucket '{DEFAULT_SKILLS_BUCKET_ID}'.")

    api.download_bucket_files(DEFAULT_SKILLS_BUCKET_ID, download_specs)


def _atomic_replace_directory(existing_dir: Path, staged_dir: Path) -> None:
    backup_dir = staged_dir.parent / f"{existing_dir.name}.backup"
    try:
        existing_dir.rename(backup_dir)
        staged_dir.rename(existing_dir)
        shutil.rmtree(backup_dir)
    except Exception:
        if backup_dir.exists() and not existing_dir.exists():
            backup_dir.rename(existing_dir)
        raise


def _iter_unique_skill_dirs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    discovered: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name.startswith("."):
                continue
            if not child.is_dir() and not child.is_symlink():
                continue
            resolved = child.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _apply_single_update(
    api, skill_dir: Path, marketplace_skills: dict[str, MarketplaceSkill], hf_cli_content: str
) -> SkillUpdateInfo:
    base = SkillUpdateInfo(name=skill_dir.name, skill_dir=skill_dir, status="unmanaged")

    if not (skill_dir / MANAGED_MARKER_FILENAME).exists():
        return base

    if skill_dir.name == DEFAULT_SKILL_ID:
        try:
            install_generated_skill(hf_cli_content, skill_dir.parent, force=True)
        except Exception as exc:
            return replace(base, status="source_unreachable", detail=str(exc))
        return replace(base, status="up_to_date")

    skill = marketplace_skills.get(skill_dir.name.lower())
    if skill is None:
        return replace(
            base,
            status="source_unreachable",
            detail=f"Skill '{skill_dir.name}' is no longer available in {DEFAULT_SKILLS_BUCKET_ID}.",
        )

    try:
        _install_marketplace_skill(api, skill, skill_dir.parent, force=True)
    except Exception as exc:
        return replace(base, status="source_unreachable", detail=str(exc))

    return replace(base, status="up_to_date")


### `hf-cli` SKILL CHECKER


# Version stamp written in the generated SKILL.md (see `build_skill_md`).
_GENERATED_VERSION_RE = re.compile(r"Generated with `huggingface_hub v([^`]+)`")


def check_skill_update() -> None:
    """Hint at installing or updating the `hf-cli` skill, at most once every 24 hours.

    Only prints a hint: installing or updating stays an explicit user action.

    Skipped entirely if `HF_HUB_DISABLE_UPDATE_CHECK` is set. Called at the entry point of
    the CLI, so any error is caught and logged to avoid breaking the CLI.
    """
    try:
        _check_skill_update()
    except Exception:
        logger.debug("Error while checking for hf-cli skill update.", exc_info=True)


def _check_skill_update() -> None:
    if constants.HF_HUB_DISABLE_UPDATE_CHECK:
        return

    # Skip if already checked in the last 24 hours
    stamp_path = Path(constants.CHECK_FOR_SKILL_UPDATE_DONE_PATH)
    if stamp_path.exists() and (time.time() - stamp_path.stat().st_mtime) < 24 * 3600:
        return
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.touch()

    global_dirs = _installed_hf_cli_dirs(constants.AGENTS_SKILLS_GLOBAL_PATH, constants.CLAUDE_SKILLS_GLOBAL_PATH)
    local_dirs = _installed_hf_cli_dirs(constants.AGENTS_SKILLS_LOCAL_PATH, constants.CLAUDE_SKILLS_LOCAL_PATH)

    if not global_dirs and not local_dirs:
        # Don't push the skill on pre-release / dev versions (same rule as the PyPI update check).
        if any(tag in __version__ for tag in ["rc", "dev"]):
            return
        out.hint(
            f"The `{DEFAULT_SKILL_ID}` skill is not installed. Run `hf skills add -g`"
            " to teach your AI agents how to use the `hf` CLI."
        )
        return

    # Suggest the command matching where the outdated skill is installed (global takes precedence).
    if any(_is_skill_outdated(skill_dir) for skill_dir in global_dirs):
        scope = " -g"
    elif any(_is_skill_outdated(skill_dir) for skill_dir in local_dirs):
        scope = ""
    else:
        return
    out.hint(
        f"The installed `{DEFAULT_SKILL_ID}` skill was generated by another `hf` version."
        f" Run `hf skills update {DEFAULT_SKILL_ID}{scope}` to refresh it."
    )


def _installed_hf_cli_dirs(*roots: Path) -> list[Path]:
    dirs = [(root.expanduser() / DEFAULT_SKILL_ID).resolve() for root in roots]
    return [skill_dir for skill_dir in dict.fromkeys(dirs) if skill_dir.is_dir()]


def _is_skill_outdated(skill_dir: Path) -> bool:
    """Whether the installed `hf-cli` SKILL.md was generated by another `hf` version."""
    try:
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return False
    match = _GENERATED_VERSION_RE.search(content)
    return match is None or match.group(1) != __version__
<|endoftext|># Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import sys
import traceback
from typing import Annotated, Any

import click

from huggingface_hub import __version__, constants
from huggingface_hub.cli._cli_utils import (
    LazyHfCommand,
    LazyHfGroup,
    check_cli_update,
    fallback_typer_group_factory,
    typer_factory,
)
from huggingface_hub.cli._output import out
from huggingface_hub.utils import logging

from ._completion import _COMPLETE_VAR, InstallCompletionOpt, ShowCompletionOpt
from ._framework import Option


def _dispatch_unknown_top_level_extension(args: list[str], built_in_commands: set[str]) -> int | None:
    from .extensions import dispatch_unknown_top_level_extension

    return dispatch_unknown_top_level_extension(args, built_in_commands)


def _list_installed_extensions_for_help() -> list[tuple[str, str]]:
    from .extensions import list_installed_extensions_for_help

    return list_installed_extensions_for_help()


def check_skill_update() -> None:
    from ._skills import check_skill_update as _check_skill_update

    _check_skill_update()


app = typer_factory(
    help="Hugging Face Hub CLI",
    cls=fallback_typer_group_factory(
        _dispatch_unknown_top_level_extension,
        extra_commands_provider=_list_installed_extensions_for_help,
    ),
)


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise click.exceptions.Exit()


def _skills_callback(value: bool) -> None:
    if value:
        from .skills import skills_preview

        skills_preview()
        raise click.exceptions.Exit()


@app.group_callback(invoke_without_command=True)
def app_callback(
    version: Annotated[
        bool | None, Option("-v", "--version", callback=_version_callback, is_eager=True, hidden=True)
    ] = None,
    skills: Annotated[
        bool,
        Option(
            "--skills",
            callback=_skills_callback,
            is_eager=True,
            help="Print the `hf-cli` SKILL.md to stdout (alias for `hf skills preview`).",
        ),
    ] = False,
    install_completion: InstallCompletionOpt = False,
    show_completion: ShowCompletionOpt = False,
) -> None:
    pass


_LAZY_COMMANDS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("sync", "buckets", "sync", {}),
    ("cp", "_cp", "make_cp", {"examples_attribute": "CP_EXAMPLES", "is_factory": True}),
    ("download", "download", "download", {"examples_attribute": "DOWNLOAD_EXAMPLES"}),
    ("upload", "upload", "upload", {"examples_attribute": "UPLOAD_EXAMPLES"}),
    ("env", "system", "env", {"topic": "help"}),
    ("update", "system", "update", {"topic": "help"}),
    ("version", "system", "version", {"topic": "help"}),
    ("lfs-enable-largefiles", "lfs", "lfs_enable_largefiles", {"hidden": True}),
    ("lfs-multipart-upload", "lfs", "lfs_multipart_upload", {"hidden": True}),
]


_LAZY_GROUPS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("auth", "auth", "auth_cli", {}),
    ("buckets", "buckets", "buckets_cli", {}),
    ("cache", "cache", "cache_cli", {}),
    ("collections", "collections", "collections_cli", {}),
    ("datasets", "datasets", "datasets_cli", {}),
    ("discussions", "discussions", "discussions_cli", {}),
    ("jobs", "jobs", "jobs_cli", {}),
    ("models", "models", "models_cli", {}),
    ("papers", "papers", "papers_cli", {}),
    ("repos", "repos", "repos_cli", {}),
    ("sandbox", "sandbox", "sandbox_cli", {}),
    ("skills", "skills", "skills_cli", {}),
    ("spaces", "spaces", "spaces_cli", {}),
    ("webhooks", "webhooks", "webhooks_cli", {}),
    ("endpoints", "inference_endpoints", "ie_cli", {}),
    ("extensions | ext", "extensions", "extensions_cli", {}),
]

for name, module, attribute, extras in _LAZY_COMMANDS:
    app.add_command(
        LazyHfCommand(
            name,
            module=f"huggingface_hub.cli.{module}",
            attribute=attribute,
            **extras,
        ),
        name,
    )

for name, module, attribute, extras in _LAZY_GROUPS:
    app.add_command(
        LazyHfGroup(
            name,
            module=f"huggingface_hub.cli.{module}",
            attribute=attribute,
            **extras,
        ),
        name,
    )


def main():
    # Shell-completion requests must stay fast and emit nothing but candidates:
    # skip the startup work and let click handle the env var inside `app()`.
    if _COMPLETE_VAR not in os.environ:
        if not constants.HF_DEBUG:
            logging.set_verbosity_info()
        check_cli_update("huggingface_hub")
        # Don't nag while the user is already managing skills, nor on `hf update` which handles the
        # skill itself (it would print a redundant or contradictory hint before doing so).
        if sys.argv[1:2] not in (["skills"], ["update"], ["--skills"]):
            check_skill_update()

    try:
        # Click glob-expands args on Windows by default, even quoted ones like `--exclude "*.bin"`.
        app(windows_expand_args=False)
    except Exception as e:
        from ._errors import format_known_exception

        message = format_known_exception(e)
        if message:
            out.error(message)
            if constants.HF_DEBUG:
                traceback.print_exc()
            else:
                out.hint("set HF_DEBUG=1 as environment variable for full traceback.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
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
"""Contains commands to interact with discussions and pull requests on the Hugging Face Hub."""

import enum
import sys
from pathlib import Path
from typing import Annotated

import click

from huggingface_hub import constants

from ._cli_utils import (
    AuthorOpt,
    LimitOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOpt,
    TokenOpt,
    get_hf_api,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


class DiscussionStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    merged = "merged"
    draft = "draft"
    all = "all"


class DiscussionKind(str, enum.Enum):
    all = "all"
    discussion = "discussion"
    pull_request = "pull_request"


# "merged" and "draft" are valid Discussion statuses but the Hub API filter
# (DiscussionStatusFilter) only accepts "all", "open", "closed". When the user
# asks for merged/draft we fetch with api_status=None (i.e. all) and filter
# client-side.
_CLIENT_SIDE_STATUSES = {"merged", "draft"}


DiscussionNumArg = Annotated[
    int,
    Argument(
        help="The discussion or pull request number.",
        min=1,
    ),
]


def _read_body(body: str | None, body_file: Path | None) -> str | None:
    """Resolve body text from --body or --body-file (supports '-' for stdin)."""
    if body is not None and body_file is not None:
        raise click.BadParameter("Cannot use both --body and --body-file.")
    if body_file is not None:
        if str(body_file) == "-":
            return sys.stdin.read()
        return body_file.read_text(encoding="utf-8")
    return body


discussions_cli = typer_factory(help="Manage discussions and pull requests on the Hub.")


@discussions_cli.command(
    "list | ls",
    examples=[
        "hf discussions list username/my-model",
        "hf discussions list username/my-model --kind pull_request --status merged",
        "hf discussions list username/my-dataset --type dataset --status closed",
        "hf discussions list username/my-model --author alice --format json",
    ],
)
def discussion_list(
    repo_id: RepoIdArg,
    status: Annotated[
        DiscussionStatus,
        Option(
            "-s",
            "--status",
            help="Filter by status (open, closed, merged, draft, all).",
        ),
    ] = DiscussionStatus.open,
    kind: Annotated[
        DiscussionKind,
        Option(
            "-k",
            "--kind",
            help="Filter by kind (discussion, pull_request, all).",
        ),
    ] = DiscussionKind.all,
    author: AuthorOpt = None,
    limit: LimitOpt = 30,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """List discussions and pull requests on a repo."""
    api = get_hf_api(token=token)

    api_status: constants.DiscussionStatusFilter | None
    if status == DiscussionStatus.open:
        api_status = "open"
    elif status == DiscussionStatus.closed:
        api_status = "closed"
    else:
        api_status = None

    api_discussion_type: constants.DiscussionTypeFilter | None
    if kind == DiscussionKind.all:
        api_discussion_type = None
    else:
        api_discussion_type = kind.value  # type: ignore[assignment]

    discussions = []
    for d in api.get_repo_discussions(
        repo_id=repo_id,
        author=author,
        discussion_type=api_discussion_type,
        discussion_status=api_status,
        repo_type=repo_type.value,
    ):
        if status.value in _CLIENT_SIDE_STATUSES and d.status != status.value:
            continue
        discussions.append(d)
        if len(discussions) >= limit:
            break

    items = [_dataclass_to_dict(d) for d in discussions]
    out.table(
        items,
        headers=["num", "title", "is_pull_request", "status", "author", "created_at"],
        id_key="num",
    )


@discussions_cli.command(
    "info",
    examples=[
        "hf discussions info username/my-model 5",
        "hf discussions info username/my-model 5 --format json",
    ],
)
def discussion_info(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Get info about a discussion or pull request."""
    api = get_hf_api(token=token)
    details = api.get_discussion_details(
        repo_id=repo_id,
        discussion_num=num,
        repo_type=repo_type.value,
    )
    out.dict(details)


@discussions_cli.command(
    "create",
    examples=[
        'hf discussions create username/my-model --title "Bug report"',
        'hf discussions create username/my-model --title "Feature request" --body "Please add X"',
        'hf discussions create username/my-model --title "Fix typo" --pull-request',
        'hf discussions create username/my-dataset --type dataset --title "Data quality issue"',
    ],
)
def discussion_create(
    repo_id: RepoIdArg,
    title: Annotated[
        str,
        Option(
            "--title",
            help="The title of the discussion or pull request.",
        ),
    ],
    body: Annotated[
        str | None,
        Option(
            "--body",
            help="The description (supports Markdown).",
        ),
    ] = None,
    body_file: Annotated[
        Path | None,
        Option(
            "--body-file",
            help="Read the description from a file. Use '-' for stdin.",
        ),
    ] = None,
    pull_request: Annotated[
        bool,
        Option(
            "--pull-request",
            "--pr",
            help="Create a pull request instead of a discussion.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Create a new discussion or pull request on a repo."""
    description = _read_body(body, body_file)
    api = get_hf_api(token=token)
    discussion = api.create_discussion(
        repo_id=repo_id,
        title=title,
        description=description,
        repo_type=repo_type.value,
        pull_request=pull_request,
    )
    kind = "pull request" if pull_request else "discussion"
    ref = f"refs/pr/{discussion.num}" if pull_request else None
    out.result(f"Created {kind} #{discussion.num} on {repo_id}", num=discussion.num, url=discussion.url, ref=ref)


@discussions_cli.command(
    "comment",
    examples=[
        'hf discussions comment username/my-model 5 --body "Thanks for reporting!"',
        'hf discussions comment username/my-model 5 --body "LGTM!"',
    ],
)
def discussion_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    body: Annotated[
        str | None,
        Option(
            "--body",
            help="The comment text (supports Markdown).",
        ),
    ] = None,
    body_file: Annotated[
        Path | None,
        Option(
            "--body-file",
            help="Read the comment from a file. Use '-' for stdin.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Comment on a discussion or pull request."""
    comment = _read_body(body, body_file)
    if comment is None:
        raise click.BadParameter("Either --body or --body-file is required.")
    api = get_hf_api(token=token)
    api.comment_discussion(
        repo_id=repo_id,
        discussion_num=num,
        comment=comment,
        repo_type=repo_type.value,
    )
    out.result(f"Commented on #{num} in {repo_id}", num=num, repo=repo_id)


@discussions_cli.command(
    "edit",
    examples=[
        'hf discussions edit username/my-model 5 abc123 --body "Updated comment."',
        "hf discussions edit username/my-model 5 abc123 --body-file fixed.md",
    ],
)
def discussion_edit(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment_id: Annotated[
        str,
        Argument(
            help="The ID of the comment to edit (see 'hf discussions info ... --format json').",
        ),
    ],
    body: Annotated[
        str | None,
        Option(
            "--body",
            help="The new comment text (supports Markdown).",
        ),
    ] = None,
    body_file: Annotated[
        Path | None,
        Option(
            "--body-file",
            help="Read the new comment from a file. Use '-' for stdin.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Edit an existing comment on a discussion or pull request."""
    new_content = _read_body(body, body_file)
    if new_content is None:
        raise click.BadParameter("Either --body or --body-file is required.")
    api = get_hf_api(token=token)
    api.edit_discussion_comment(
        repo_id=repo_id,
        discussion_num=num,
        comment_id=comment_id,
        new_content=new_content,
        repo_type=repo_type.value,
    )
    out.result(f"Edited comment {comment_id} on #{num} in {repo_id}", num=num, repo=repo_id, comment_id=comment_id)


@discussions_cli.command(
    "close",
    examples=[
        "hf discussions close username/my-model 5",
        'hf discussions close username/my-model 5 --comment "Closing as resolved."',
    ],
)
def discussion_close(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        str | None,
        Option(
            "--comment",
            help="An optional comment to post when closing.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Close a discussion or pull request."""
    out.confirm(f"Close #{num} on '{repo_id}'?", yes=yes)
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="closed",
        comment=comment,
        repo_type=repo_type.value,
    )
    out.result(f"Closed #{num} in {repo_id}", num=num, repo=repo_id)


@discussions_cli.command(
    "reopen",
    examples=[
        "hf discussions reopen username/my-model 5",
        'hf discussions reopen username/my-model 5 --comment "Reopening for further investigation."',
    ],
)
def discussion_reopen(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        str | None,
        Option(
            "--comment",
            help="An optional comment to post when reopening.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Reopen a closed discussion or pull request."""
    out.confirm(f"Reopen #{num} on '{repo_id}'?", yes=yes)
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="open",
        comment=comment,
        repo_type=repo_type.value,
    )
    out.result(f"Reopened #{num} in {repo_id}", num=num, repo=repo_id)


@discussions_cli.command(
    "rename",
    examples=[
        'hf discussions rename username/my-model 5 "Updated title"',
    ],
)
def discussion_rename(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    new_title: Annotated[
        str,
        Argument(
            help="The new title.",
        ),
    ],
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Rename a discussion or pull request."""
    api = get_hf_api(token=token)
    api.rename_discussion(
        repo_id=repo_id,
        discussion_num=num,
        new_title=new_title,
        repo_type=repo_type.value,
    )
    out.result(f"Renamed #{num} in {repo_id}", num=num, repo=repo_id, title=new_title)


@discussions_cli.command(
    "merge",
    examples=[
        "hf discussions merge username/my-model 5",
        'hf discussions merge username/my-model 5 --comment "Merging, thanks!"',
    ],
)
def discussion_merge(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        str | None,
        Option(
            "--comment",
            help="An optional comment to post when merging.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Merge a pull request."""
    out.confirm(f"Merge #{num} on '{repo_id}'?", yes=yes)
    api = get_hf_api(token=token)
    api.merge_pull_request(
        repo_id=repo_id,
        discussion_num=num,
        comment=comment,
        repo_type=repo_type.value,
    )
    out.result(f"Merged #{num} in {repo_id}", num=num, repo=repo_id)


@discussions_cli.command(
    "diff",
    examples=[
        "hf discussions diff username/my-model 5",
    ],
)
def discussion_diff(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Show the diff of a pull request."""
    api = get_hf_api(token=token)
    details = api.get_discussion_details(
        repo_id=repo_id,
        discussion_num=num,
        repo_type=repo_type.value,
    )
    if details.diff:
        out.text(details.diff)
    else:
        out.text("No diff available.")
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
"""Contains command to upload a repo or file with the CLI."""

import os
import time
import warnings
from typing import Annotated

import click

from huggingface_hub import constants, logging
from huggingface_hub._commit_scheduler import CommitScheduler
from huggingface_hub.errors import CLIError, RevisionNotFoundError
from huggingface_hub.utils import parse_hf_uri

from ._cli_utils import (
    PrivateOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOptionalOpt,
    RevisionOpt,
    TokenOpt,
    get_hf_api,
)
from ._framework import Argument, Option
from ._output import out


logger = logging.get_logger(__name__)


UPLOAD_EXAMPLES = [
    "hf upload my-cool-model . .",
    "hf upload Wauplin/my-cool-model ./models/model.safetensors",
    "hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset",
    'hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%"',
    "hf upload bigcode/the-stack . . --repo-type dataset --create-pr",
]


def upload(
    repo_id: RepoIdArg,
    local_path: Annotated[
        str | None,
        Argument(
            help="Local path to the file or folder to upload. Wildcard patterns are supported. Defaults to current directory.",
        ),
    ] = None,
    path_in_repo: Annotated[
        str | None,
        Argument(
            help="Path of the file or folder in the repo. Defaults to the relative path of the file or folder.",
        ),
    ] = None,
    repo_type: RepoTypeOptionalOpt = None,
    revision: RevisionOpt = None,
    private: PrivateOpt = None,
    include: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to match files to upload.",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to exclude from files to upload.",
        ),
    ] = None,
    delete: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns for file to be deleted from the repo while committing.",
        ),
    ] = None,
    commit_message: Annotated[
        str | None,
        Option(
            help="The summary / title / first line of the generated commit.",
        ),
    ] = None,
    commit_description: Annotated[
        str | None,
        Option(
            help="The description of the generated commit.",
        ),
    ] = None,
    create_pr: Annotated[
        bool,
        Option(
            help="Whether to upload content as a new Pull Request.",
        ),
    ] = False,
    every: Annotated[
        float | None,
        Option(
            help="If set, a background job is scheduled to create commits every `every` minutes.",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Upload a file or a folder to the Hub.

    Folders with many files are split into several commits. If a folder upload is interrupted, re-run the same command
    to resume it.
    """

    if every is not None and every <= 0:
        raise click.BadParameter("--every must be a positive value", param_hint="every")

    # `repo_id` may be a plain repo id or an `hf://` URI (e.g. `hf://datasets/my-org/my-dataset@v1.0/data/`).
    # When a URI is provided, it is authoritative for the repo type, revision and (optionally) path in repo,
    # so explicit `--repo-type` / `--revision` options are forbidden alongside it.
    # We branch on the `hf://` prefix (the user's *intent*) rather than on whether the string parses as a
    # valid URI: a malformed URI then surfaces a precise `HfUriError` (formatted globally in `cli/_errors.py`)
    # instead of silently falling through to the plain-repo-id path and failing later with an opaque error.
    if repo_id.startswith(constants.HF_PROTOCOL):
        if repo_type is not None:
            raise CLIError(f"'--repo-type' cannot be used with an 'hf://' URI ('{repo_id}').")
        if revision is not None:
            raise CLIError(f"'--revision' cannot be used with an 'hf://' URI ('{repo_id}').")
        uri = parse_hf_uri(repo_id)
        if uri.is_bucket:
            raise CLIError("Buckets are not supported by `hf upload`. Use `hf sync` instead.")
        repo_id, repo_type_str, revision = uri.id, uri.type, uri.revision
        if uri.path_in_repo:
            if path_in_repo is not None:
                raise CLIError(
                    f"Cannot combine a path in the hf:// URI ('{uri.path_in_repo}') with the `path_in_repo` argument ('{path_in_repo}')."
                )
            path_in_repo = uri.path_in_repo
    else:
        repo_type_str = (repo_type or RepoType.model).value

    api = get_hf_api(token=token)

    # Resolve local_path and path_in_repo based on implicit/explicit rules
    resolved_local_path, resolved_path_in_repo, resolved_include = _resolve_upload_paths(
        repo_id=repo_id, local_path=local_path, path_in_repo=path_in_repo, include=include
    )

    def run_upload() -> str:
        if os.path.isfile(resolved_local_path):
            if resolved_include is not None and len(resolved_include) > 0 and isinstance(resolved_include, list):
                warnings.warn("Ignoring --include since a single file is uploaded.")
            if exclude is not None and len(exclude) > 0:
                warnings.warn("Ignoring --exclude since a single file is uploaded.")
            if delete is not None and len(delete) > 0:
                warnings.warn("Ignoring --delete since a single file is uploaded.")

        # Schedule commits if `every` is set
        if every is not None:
            allow_patterns: list[str] | None
            ignore_patterns: list[str] | None
            if os.path.isfile(resolved_local_path):
                # If file => watch entire folder + use allow_patterns.
                # `CommitScheduler` matches patterns against paths relative to `folder_path` and
                # re-uploads them under `path_in_repo`, so the pattern is the file name and only
                # the directory part of `path_in_repo` can be honoured.
                folder_path = os.path.dirname(resolved_local_path)
                filename = os.path.basename(resolved_local_path)
                pi, _, dest_filename = resolved_path_in_repo.rpartition("/")
                # An empty or "." destination name means a directory, not a rename.
                if dest_filename not in ("", ".", filename):
                    out.warning(
                        f"Scheduled uploads keep the local file name, so the file will be uploaded as"
                        f" '{(pi + '/' if pi else '') + filename}' rather than '{resolved_path_in_repo}'."
                        " Rename the local file to upload it under a different name."
                    )
                allow_patterns = [filename]
                ignore_patterns = []
            else:
                folder_path = resolved_local_path
                pi = resolved_path_in_repo
                allow_patterns = resolved_include
                ignore_patterns = exclude
                if delete is not None and len(delete) > 0:
                    warnings.warn("Ignoring --delete when uploading with scheduled commits.")

            scheduler = CommitScheduler(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type=repo_type_str,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                path_in_repo=pi,
                private=private,
                every=every,
                hf_api=api,
            )
            out.text(f"Scheduling commits every {every} minutes to {scheduler.repo_id}.")
            try:
                while True:
                    time.sleep(100)
            except KeyboardInterrupt:
                scheduler.stop()
                return "Stopped scheduled commits."

        # Otherwise, create repo and proceed with the upload
        if not os.path.isfile(resolved_local_path) and not os.path.isdir(resolved_local_path):
            raise FileNotFoundError(f"No such file or directory: '{resolved_local_path}'.")
        created = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type_str,
            exist_ok=True,
            private=private,
            space_sdk="gradio" if repo_type_str == "space" else None,
            # ^ We don't want it to fail when uploading to a Space => let's set Gradio by default.
            # ^ I'd rather not add CLI args to set it explicitly as we already have `hf repos create` for that.
        ).repo_id

        # Check if branch already exists and if not, create it
        if revision is not None and not create_pr:
            try:
                api.repo_info(repo_id=created, repo_type=repo_type_str, revision=revision)
            except RevisionNotFoundError:
                logger.info(f"Branch '{revision}' not found. Creating it...")
                api.create_branch(repo_id=created, repo_type=repo_type_str, branch=revision, exist_ok=True)
                # ^ `exist_ok=True` to avoid race concurrency issues

        # File-based upload
        if os.path.isfile(resolved_local_path):
            return api.upload_file(
                path_or_fileobj=resolved_local_path,
                path_in_repo=resolved_path_in_repo,
                repo_id=created,
                repo_type=repo_type_str,
                revision=revision,
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
            )

        # Folder-based upload
        return api.upload_folder(
            folder_path=resolved_local_path,
            path_in_repo=resolved_path_in_repo,
            repo_id=created,
            repo_type=repo_type_str,
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            allow_patterns=resolved_include,
            ignore_patterns=exclude,
            delete_patterns=delete,
        )

    result = run_upload()
    out.result("Uploaded", url=result)


def _resolve_upload_paths(
    *, repo_id: str, local_path: str | None, path_in_repo: str | None, include: list[str] | None
) -> tuple[str, str, list[str] | None]:
    repo_name = repo_id.split("/")[-1]
    resolved_include = include

    # An existing path is never a wildcard, even if it contains glob characters (e.g. "model [v2].safetensors")
    if local_path is not None and not os.path.exists(local_path) and any(c in local_path for c in ["*", "?", "["]):
        if include is not None:
            raise ValueError("Cannot set --include when local_path contains a wildcard.")
        if path_in_repo is not None and path_in_repo != ".":
            raise ValueError("Cannot set path_in_repo when local_path contains a wildcard.")
        return ".", ".", [local_path]

    if local_path is None and os.path.isfile(repo_name):
        return repo_name, repo_name, resolved_include
    if local_path is None and os.path.isdir(repo_name):
        return repo_name, ".", resolved_include
    if local_path is None:
        raise ValueError(f"'{repo_name}' is not a local file or folder. Please set local_path explicitly.")

    if path_in_repo is None and os.path.isfile(local_path):
        return local_path, os.path.basename(local_path), resolved_include
    if path_in_repo is None:
        return local_path, ".", resolved_include
    return local_path, path_in_repo, resolved_include
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
"""Contains commands to run and manage sandboxes on Hugging Face Jobs."""

import sys
import time
from contextlib import contextmanager
from typing import Annotated, Any, Iterator

import click

from huggingface_hub._sandbox import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_IMAGE,
    DEFAULT_SANDBOXES_PER_HOST,
    MODE_LABEL,
    MODE_POOL,
    POOL_LABEL,
    SANDBOX_LABEL,
    SHARED_ID_SEP,
    Sandbox,
    SandboxPool,
    SandboxProcess,
    _split_sandbox_id,
)
from huggingface_hub._sandbox_cache import delete_pool_cache
from huggingface_hub.errors import CLIError, SandboxError

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    SecretsFileOpt,
    SecretsOpt,
    TokenOpt,
    VolumesOpt,
    get_hf_api,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import out
from .jobs import FlavorOpt, LabelsOpt, NamespaceOpt, _parse_labels_map


sandbox_cli = typer_factory(help="Run and manage experimental sandboxes on Hugging Face Jobs.")
pool_cli = typer_factory(
    help="Warm host VM pools and spawn experimental shared sandboxes for workloads within the same trust boundary."
)
sandbox_cli.add_group(pool_cli, name="pool")
process_cli = typer_factory(help="List and stop background processes running in a sandbox.")
sandbox_cli.add_group(process_cli, name="process")

SandboxIdArg = Annotated[str, Argument(help="The sandbox id as printed by `hf sandbox create`.")]


@contextmanager
def _connect(sandbox_id: str, *, namespace: str | None, token: str | None) -> Iterator[Sandbox]:
    """Reattach to a sandbox and close the HTTP client when the command is done."""
    sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
    try:
        yield sandbox
    finally:
        sandbox.close()


@sandbox_cli.command(
    "create",
    examples=[
        "hf sandbox create",
        "hf sandbox create ubuntu:24.04",
        "hf sandbox create --flavor a10g-small",
        "hf sandbox create --label controller-run=run-42",
        "hf sandbox create --pool pool-ab12cd34ef56 --env LOG_LEVEL=debug",
    ],
)
def sandbox_create(
    image: Annotated[str | None, Argument(help="Docker image (needs /bin/sh).")] = None,
    pool: Annotated[
        str | None,
        Option("--pool", help="Spawn a cheap shared sandbox in this pool (from `hf sandbox pool create`)."),
    ] = None,
    flavor: FlavorOpt = None,
    idle_timeout: Annotated[
        str | None,
        Option(help="Auto-terminate the sandbox after this much inactivity (e.g. '10m'). Defaults to 10m."),
    ] = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    label: LabelsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    volume: VolumesOpt = None,
    namespace: NamespaceOpt = None,
    forward_hf_token: Annotated[
        bool, Option("--forward-hf-token", help="Inject your HF token as HF_TOKEN in the sandbox.")
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Create a sandbox: a dedicated VM by default, or a cheap shared one with `--pool`.

    Env and idle-timeout apply to the sandbox in both modes. With `--pool`, the image and
    flavor come from the pool, so passing them here is an error; `--secrets` is also
    rejected since pooled sandboxes have no encrypted-secrets channel (use `--env`). Define
    a pool first with `hf sandbox pool create`.
    """
    start = time.time()
    idle = idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT

    if pool is not None:
        if image is not None or flavor is not None or volume:
            raise CLIError("--pool fixes the image/flavor (and volumes aren't supported); drop those options.")
        if secrets or secrets_file:
            raise CLIError(
                "--pool has no encrypted-secrets channel. Pass the values with --env/--env-file: on a pooled "
                "sandbox they are delivered to the host at creation and are not stored in the job metadata, "
                "but they are not encrypted at rest."
            )
        if label:
            raise CLIError("--label is only supported for dedicated sandboxes.")
        sbx = SandboxPool.connect(pool, namespace=namespace, token=token).create(
            env=parse_env_map(env, env_file),
            idle_timeout=idle,
            forward_hf_token=forward_hf_token,
        )
        out.result("Sandbox ready", id=sbx.id, host=sbx.host_id, pool=pool, elapsed=f"{time.time() - start:.1f}s")
        out.hint(f"Run a command with `hf sandbox exec {sbx.id} -- echo hello`.")
        out.hint(f"Terminate it with `hf sandbox kill {sbx.id}`.")
        return

    sandbox = Sandbox.create(
        image=image or DEFAULT_IMAGE,
        flavor=flavor or "cpu-basic",
        idle_timeout=idle,
        env=parse_env_map(env, env_file),
        secrets=parse_env_map(secrets, secrets_file),
        volumes=parse_volumes(volume),
        namespace=namespace,
        forward_hf_token=forward_hf_token,
        labels=_parse_labels_map(label),
        token=token,
    )
    # Release the HTTP client (the sandbox keeps running)
    sandbox.close()
    out.result("Sandbox ready", id=sandbox.id, image=sandbox.image, elapsed=f"{time.time() - start:.1f}s")
    out.hint(f"Run a command with `hf sandbox exec {sandbox.id} -- echo hello`.")
    out.hint(f"Terminate it with `hf sandbox kill {sandbox.id}`.")


@sandbox_cli.command(
    "exec",
    context_settings={"ignore_unknown_options": True},
    examples=[
        'hf sandbox exec <sandbox_id> -- python -c "print(42)"',
        "hf sandbox exec -w /app <sandbox_id> -- pytest -x",
    ],
)
def sandbox_exec(
    sandbox_id: SandboxIdArg,
    command: Annotated[list[str], Argument(help="The command to run.")],
    workdir: Annotated[str | None, Option("-w", "--workdir", help="Working directory.")] = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    exec_timeout: Annotated[
        float | None, Option("--timeout", help="Kill the command after this many seconds.")
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Run a command in a sandbox, streaming output. Exits with the command's exit code.

    To start a long-running command in the background instead of waiting for it, use
    `hf sandbox spawn`.
    """

    def write_stdout(data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()

    def write_stderr(data: str) -> None:
        sys.stderr.write(data)
        sys.stderr.flush()

    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        result = sandbox.run(
            list(command),
            env=parse_env_map(env, env_file),
            cwd=workdir,
            timeout=exec_timeout,
            on_stdout=write_stdout,
            on_stderr=write_stderr,
            capture_output=False,
            check=False,
        )
    if result.timed_out:
        out.error(f"Command timed out after {exec_timeout}s.")
        raise click.exceptions.Exit(code=result.exit_code or 124)  # 124: conventional timeout exit code
    if result.exit_code != 0:
        raise click.exceptions.Exit(code=result.exit_code if result.exit_code is not None else 1)


@sandbox_cli.command(
    "spawn",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "hf sandbox spawn <sandbox_id> -- python -m http.server 8000",
        "hf sandbox spawn -w /app <sandbox_id> -- uvicorn app:app",
    ],
)
def sandbox_spawn(
    sandbox_id: SandboxIdArg,
    command: Annotated[list[str], Argument(help="The command to run in the background.")],
    workdir: Annotated[str | None, Option("-w", "--workdir", help="Working directory.")] = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Start a long-running command in the background and return its pid (don't wait).

    List a sandbox's processes with `hf sandbox process ls` and stop one with
    `hf sandbox process kill`.
    """
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        process = sandbox.run(list(command), env=parse_env_map(env, env_file), cwd=workdir, background=True)
    out.result("Process started", sandbox=sandbox_id, pid=process.pid)
    out.hint(f"List processes with `hf sandbox process ls {sandbox_id}`.")
    out.hint(f"Stop it with `hf sandbox process kill {sandbox_id} {process.pid}`.")


@sandbox_cli.command(
    "cp",
    examples=[
        "hf sandbox cp data.csv <sandbox_id>:/data/data.csv",
        "hf sandbox cp <sandbox_id>:/app/result.json result.json",
    ],
)
def sandbox_cp(
    src: Annotated[str, Argument(help="Source: a local path or <sandbox_id>:<path>.")],
    dst: Annotated[str, Argument(help="Destination: a local path or <sandbox_id>:<path>.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Copy a file between the local machine and a sandbox (docker-style)."""

    def parse(ref: str) -> tuple[str | None, str]:
        # Only treat as a sandbox ref when the part before ':' looks like a sandbox id
        # (more than one char): this leaves local paths and Windows drive letters like
        # 'C:\data\file.csv' or 'C:/data/file.csv' (single-letter prefix) untouched.
        if ":" in ref and not ref.startswith((".", "/", "~")):
            sandbox_id, path = ref.split(":", 1)
            if len(sandbox_id) > 1:
                return sandbox_id, path
        return None, ref

    src_sandbox, src_path = parse(src)
    dst_sandbox, dst_path = parse(dst)
    if (src_sandbox is None) == (dst_sandbox is None):
        raise CLIError("Exactly one of SRC and DST must be a sandbox path (<sandbox_id>:<path>).")
    if src_sandbox is not None:
        with _connect(src_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.download(src_path, dst_path)
    else:
        assert dst_sandbox is not None
        with _connect(dst_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.upload(src_path, dst_path)
    out.result("Copied", src=src, dst=dst)


@sandbox_cli.command(
    "kill",
    examples=[
        "hf sandbox kill <sandbox_id>",
        "hf sandbox kill <host_id>   # kills a whole shared host (all its sandboxes)",
        "hf sandbox kill --all",
    ],
)
def sandbox_kill(
    sandbox_id: Annotated[str | None, Argument(help="The sandbox or host id to terminate.")] = None,
    all_: Annotated[bool, Option("--all", help="Terminate every sandbox and host in the namespace.")] = False,
    yes: Annotated[bool, Option("-y", "--yes", help="Answer Yes to prompts automatically.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Terminate a sandbox, a whole shared host, or everything (--all)."""
    api = get_hf_api(token=token)

    if all_:
        jobs = list(api.list_jobs(status=["RUNNING", "SCHEDULING"], labels={SANDBOX_LABEL: "1"}, namespace=namespace))
        if not jobs:
            out.text("No running sandboxes.")
            return
        out.confirm(f"Terminate {len(jobs)} sandbox job(s) (including shared hosts and all their sandboxes)?", yes=yes)
        for job in jobs:
            api.cancel_job(job_id=job.id, namespace=job.owner.name)
        out.result("Terminated", jobs=len(jobs))
        return

    if sandbox_id is None:
        raise CLIError("Provide a sandbox id, a host id, or --all.")

    sid, ns = _split_sandbox_id(sandbox_id, namespace)
    if SHARED_ID_SEP in sid:
        # One shared sandbox: remove it from its host (frees a slot; host keeps running).
        try:
            with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
                sandbox.kill()
        except SandboxError as e:
            raise CLIError(str(e)) from e
        out.result("Sandbox terminated", id=sandbox_id)
        return

    # A bare id: either a dedicated sandbox job or a shared host job.
    job = api.inspect_job(job_id=sid, namespace=ns)
    if (job.labels or {}).get(MODE_LABEL) == MODE_POOL:
        out.confirm(f"Terminate shared host {sid} and all of its sandboxes?", yes=yes)
        api.cancel_job(job_id=job.id, namespace=job.owner.name)
        out.result("Host terminated", id=sid)
        return
    try:
        with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
            sandbox.kill()
    except SandboxError as e:
        raise CLIError(str(e)) from e
    out.result("Sandbox terminated", id=sandbox_id)


@pool_cli.command(
    "create",
    examples=[
        "hf sandbox pool create",
        "hf sandbox pool create python:3.12 --flavor cpu-basic",
        "hf sandbox pool create --per-host 50 --idle-timeout 30m",
    ],
)
def pool_create(
    image: Annotated[str | None, Argument(help="Docker image for the hosts (needs /bin/sh).")] = None,
    flavor: FlavorOpt = None,
    per_host: Annotated[
        int,
        Option("--per-host", min=1, help=f"Sandboxes packed per host VM (default {DEFAULT_SANDBOXES_PER_HOST})."),
    ] = DEFAULT_SANDBOXES_PER_HOST,
    max_hosts: Annotated[
        int | None, Option("--max-hosts", min=1, help="Optional cap on the number of host VMs.")
    ] = None,
    idle_timeout: Annotated[
        str | None,
        Option(help="Shut a host down once it has had no sandboxes for this long (e.g. '10m'). Defaults to 10m."),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Warm a pool: boot one host VM now, tagged so it can be found later by its pool id."""
    start = time.time()
    image = image or DEFAULT_IMAGE
    pool = SandboxPool(
        image=image,
        flavor=flavor or "cpu-basic",
        sandboxes_per_host=per_host,
        max_hosts=max_hosts,
        idle_timeout=idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,
        namespace=namespace,
        token=token,
    )
    pool_id = pool.name
    host_ids = pool.host_ids
    out.result(
        "Pool created",
        id=pool_id,
        image=image,
        flavor=flavor or "cpu-basic",
        host=host_ids[0],
        elapsed=f"{time.time() - start:.1f}s",
    )
    out.hint(f"Spawn a sandbox with `hf sandbox create --pool {pool_id}`.")
    out.hint(f"Delete the pool (and its hosts) with `hf sandbox pool delete {pool_id}`.")


@pool_cli.command("ls | list", examples=["hf sandbox pool ls"])
def pool_ls(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List running sandbox pools (grouped from their host VMs)."""
    api = get_hf_api(token=token)
    pools: dict[str, dict[str, Any]] = {}
    for job in api.list_jobs(status="RUNNING", labels={MODE_LABEL: MODE_POOL}, namespace=namespace):
        pid = (job.labels or {}).get(POOL_LABEL)
        if not pid:
            continue
        env = job.environment if isinstance(job.environment, dict) else {}
        info = pools.setdefault(
            pid,
            {
                "id": pid,
                "image": job.docker_image or job.space_id,
                "flavor": job.flavor,
                "per_host": env.get("SBX_CAPACITY", ""),
                "hosts": 0,
            },
        )
        info["hosts"] += 1
    rows = list(pools.values())
    out.table(rows, id_key="id")
    if not rows:
        out.hint("Create one with `hf sandbox pool create`.")
    else:
        out.hint("Spawn a sandbox with `hf sandbox create --pool <id>`.")


@pool_cli.command(
    "delete | rm",
    examples=["hf sandbox pool delete <pool_id>"],
)
def pool_delete(
    pool_id: Annotated[str, Argument(help="Pool id to delete.")],
    yes: Annotated[bool, Option("-y", "--yes", help="Answer Yes to prompts automatically.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Terminate every host VM of a pool (and therefore all its sandboxes)."""
    api = get_hf_api(token=token)
    hosts = list(
        api.list_jobs(
            status=["RUNNING", "SCHEDULING"],
            labels={MODE_LABEL: MODE_POOL, POOL_LABEL: pool_id},
            namespace=namespace,
        )
    )
    if not hosts:
        delete_pool_cache(pool_id)
        out.text(f"No running hosts for pool '{pool_id}'.")
        return
    out.confirm(f"Terminate {len(hosts)} host(s) of pool '{pool_id}' (and all their sandboxes)?", yes=yes)
    for job in hosts:
        api.cancel_job(job_id=job.id, namespace=job.owner.name)
    delete_pool_cache(pool_id)
    out.result("Pool deleted", id=pool_id, hosts_terminated=len(hosts))


def _fmt_cmd(cmd: str | list[str]) -> str:
    return cmd if isinstance(cmd, str) else " ".join(cmd)


def _fmt_status(process: SandboxProcess) -> str:
    if process.running:
        return "running"
    return "exited" if process.exit_code is None else f"exited ({process.exit_code})"


@process_cli.command("ls | list", examples=["hf sandbox process ls <sandbox_id>"])
def process_ls(
    sandbox_id: SandboxIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List the background processes running in a sandbox (started with `hf sandbox spawn`)."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        processes = sandbox.processes()
    rows = [{"pid": p.pid, "status": _fmt_status(p), "cmd": _fmt_cmd(p.cmd)} for p in processes]
    out.table(rows, id_key="pid")
    if not rows:
        out.hint(f"Start one with `hf sandbox spawn {sandbox_id} -- <cmd>`.")
    else:
        out.hint(f"Stop one with `hf sandbox process kill {sandbox_id} <pid>`.")


@process_cli.command("kill", examples=["hf sandbox process kill <sandbox_id> <pid>"])
def process_kill(
    sandbox_id: SandboxIdArg,
    pid: Annotated[int, Argument(help="The pid as printed by `hf sandbox process ls`.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Stop a background process running in a sandbox."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        process = next((p for p in sandbox.processes() if p.pid == pid), None)
        if process is None:
            raise CLIError(f"No process with pid {pid} in sandbox {sandbox_id}.")
        process.kill()
    out.result("Process stopped", sandbox=sandbox_id, pid=pid)
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
"""Read launch defaults from a UV script's optional `[tool.hf-jobs]` PEP 723 table.

See the CLI guide for the header format and `cli/jobs.py` for the merge rules.
"""

import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx2

from huggingface_hub import constants
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import SoftTemporaryDirectory, get_session


# Reference regex from PEP 723 (https://peps.python.org/pep-0723/#reference-implementation)
_PEP_723_REGEX = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")

TABLE_NAME = "tool.hf-jobs"


@dataclass
class UvScriptHeader:
    """The `[tool.hf-jobs]` table of a UV script: the launch configuration the script carries."""

    image: str | None = None
    flavor: str | None = None
    python: str | None = None
    timeout: str | None = None
    name: str | None = None
    namespace: str | None = None
    network_group: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    network_aliases: list[str] = field(default_factory=list)


VALID_KEYS = tuple(f.name for f in fields(UvScriptHeader))


@dataclass
class UvScript:
    """A UV script ready to be submitted, with the launch config it carries (if any)."""

    script: str
    """Path (or command) to pass to the Jobs API. A URL script is downloaded, so this is a local path."""

    header: UvScriptHeader | None = None
    """Parsed `[tool.hf-jobs]` table, or `None` if the script has none."""


@contextmanager
def load_uv_script(script: str) -> Generator[UvScript, None, None]:
    """Resolve a `hf jobs uv run` script argument and read its `[tool.hf-jobs]` header.

    - a local file is read in place
    - an `http(s)` URL is downloaded to a temporary directory and from there shipped to the Job like
      any local script: the URL is fetched exactly once, at submit time, instead of being fetched
      again by the container (the header has to be read client-side anyway, since `image`, `flavor`,
      `secrets`, ... are all decided before the container exists). The download lives until the
      `with` block exits, i.e. until the Job is submitted.
    - anything else is a command (e.g. `lighteval`) and carries no header
    """
    # Scripts are read as UTF-8, not with the locale encoding: a downloaded script is written as raw
    # bytes and a header with non-ASCII values must not fail to decode depending on the platform.
    if script.startswith(("http://", "https://")):
        with SoftTemporaryDirectory(prefix="hf-jobs-uv-") as tmp_dir:
            local_path = _download_script(script, Path(tmp_dir))
            yield UvScript(
                script=str(local_path), header=parse_uv_script_header(local_path.read_text(encoding="utf-8"))
            )
        return
    path = Path(script)
    if path.is_file():
        yield UvScript(script=script, header=parse_uv_script_header(path.read_text(encoding="utf-8")))
        return
    yield UvScript(script=script)


def parse_uv_script_header(text: str) -> UvScriptHeader | None:
    """Parse the `[tool.hf-jobs]` table out of a script's PEP 723 header. Returns `None` if absent."""
    content = _extract_pep_723_block(text)
    # Cheap pre-check: scripts that don't use the feature never need a TOML parser.
    if content is None or "hf-jobs" not in content:
        return None
    table = _load_toml(content).get("tool", {}).get("hf-jobs")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise CLIError(f"'{TABLE_NAME}' must be a table in the script's PEP 723 header (i.e. `[{TABLE_NAME}]`).")

    if unknown := sorted(key for key in table if key not in VALID_KEYS):
        raise CLIError(
            f"Unknown key(s) {', '.join(repr(key) for key in unknown)} in the script's [{TABLE_NAME}] table."
            f" Valid keys are: {', '.join(VALID_KEYS)}."
        )
    return UvScriptHeader(
        image=_as_str(table, "image"),
        flavor=_as_str(table, "flavor"),
        python=_as_str(table, "python"),
        timeout=_as_timeout(table),
        name=_as_str(table, "name"),
        namespace=_as_str(table, "namespace"),
        network_group=_as_str(table, "network_group"),
        env=_as_str_table(table, "env"),
        secrets=_as_secret_names(table),
        labels=_as_str_table(table, "labels"),
        volumes=_as_str_list(table, "volumes"),
        network_aliases=_as_str_list(table, "network_aliases"),
    )


def _download_script(url: str, dest_dir: Path) -> Path:
    """Download a remote UV script into `dest_dir` and return its local path."""
    name = Path(urlsplit(url).path).name
    local_path = dest_dir / (name if name.endswith(".py") else "script.py")
    try:
        # Timeout so that an unreachable URL fails the CLI instead of hanging it.
        response = get_session().get(url, timeout=constants.DEFAULT_REQUEST_TIMEOUT)
        response.raise_for_status()
    except httpx2.HTTPError as e:
        raise CLIError(f"Could not download the UV script from '{url}': {e}") from e
    local_path.write_bytes(response.content)
    return local_path


def _extract_pep_723_block(text: str) -> str | None:
    """Return the content of the first PEP 723 `script` block, with the `# ` prefixes stripped."""
    # The PEP 723 regex anchors on '\n': normalize line endings first, otherwise a script written on
    # Windows (or fetched over HTTP) silently parses as "no header at all".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for match in _PEP_723_REGEX.finditer(text):
        if match.group("type") == "script":
            return "".join(
                line[2:] if line.startswith("# ") else line[1:]
                for line in match.group("content").splitlines(keepends=True)
            )
    return None


def _load_toml(content: str) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # tomllib was added in Python 3.11
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ImportError as e:
            raise CLIError(
                f"Reading the [{TABLE_NAME}] table of a UV script requires a TOML parser."
                " Upgrade to Python 3.11+ or install one with `pip install tomli`."
            ) from e
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise CLIError(f"Invalid TOML in the script's PEP 723 header: {e}") from e


def _type_error(key: str, value: Any, expected: str) -> CLIError:
    return CLIError(f"'{key}' in the script's [{TABLE_NAME}] table must be {expected}, got: {value!r}.")


def _as_str(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _type_error(key, value, "a string")
    return value


def _as_timeout(table: dict[str, Any]) -> str | None:
    value = table.get("timeout")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise _type_error("timeout", value, 'a duration string (e.g. "30m") or a number of seconds')
    return str(value)


def _as_str_table(table: dict[str, Any], key: str) -> dict[str, str]:
    value = table.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _type_error(key, value, "a table of strings")
    for name, item in value.items():
        # Stringifying a bool or an int here would send a value the author never wrote.
        if not isinstance(item, str):
            raise _type_error(f"{key}.{name}", item, "a string")
    return value


def _as_str_list(table: dict[str, Any], key: str) -> list[str]:
    value = table.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _type_error(key, value, "a list of strings")
    return value


def _as_secret_names(table: dict[str, Any]) -> list[str]:
    if isinstance(table.get("secrets"), dict):
        raise CLIError(
            f"'secrets' in the script's [{TABLE_NAME}] table must be a list of names (e.g. `secrets = [\"HF_TOKEN\"]`),"
            " not a table: scripts travel publicly, so secret values are always read from the environment of"
            " whoever runs the script."
        )
    return _as_str_list(table, "secrets")
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
"""Shell completion for the ``hf`` CLI, built on Click's native completion.

Click generates completion scripts activated by the ``_HF_COMPLETE`` env var (this
works out of the box because ``hf`` is a Click command). This module exposes the two
conveniences Typer used to provide — ``--install-completion`` / ``--show-completion``
— as thin eager options over that machinery.
"""

import os
from pathlib import Path
from typing import Annotated

import click
from click.shell_completion import get_completion_class

from ._framework import Option


_COMPLETE_VAR = "_HF_COMPLETE"

# Shells with a shared rc file: the line that activates completion (appended once).
_RC_ACTIVATION: dict[str, tuple[Path, str]] = {
    "bash": (Path.home() / ".bashrc", f'eval "$({_COMPLETE_VAR}=bash_source hf)"'),
    "zsh": (Path.home() / ".zshrc", f'eval "$({_COMPLETE_VAR}=zsh_source hf)"'),
}
# Fish auto-loads per-command files from its completions directory. The file is dedicated
# to `hf`, so the full script is written there directly (overwriting any stale version)
# instead of re-generating it via `hf` on every shell startup.
_FISH_COMPLETION_PATH = Path.home() / ".config" / "fish" / "completions" / "hf.fish"


def _detect_shell() -> str:
    return Path(os.environ.get("SHELL", "")).name or "bash"


def _completion_script(shell: str) -> str:
    # Imported lazily to avoid a circular import (hf.py imports the options below).
    from .hf import app

    completion_cls = get_completion_class(shell)
    if completion_cls is None:
