            return SyncOperation(action="skip", reason=skip_reason, **base_kwargs)
    elif ignore_times:
        if size_differs:
            return SyncOperation(action=action, reason="size differs", bucket_file=bucket_file, **base_kwargs)
        else:
            return SyncOperation(action="skip", reason="same size", **base_kwargs)
    else:
        if size_differs or source_newer:
            reason = "size differs" if size_differs else source_newer_label
            return SyncOperation(action=action, reason=reason, bucket_file=bucket_file, **base_kwargs)
        else:
            return SyncOperation(action="skip", reason="identical", **base_kwargs)


def _compute_sync_plan(
    source: str,
    dest: str,
    api: "HfApi",
    delete: bool = False,
    ignore_times: bool = False,
    ignore_sizes: bool = False,
    existing: bool = False,
    ignore_existing: bool = False,
    filter_matcher: FilterMatcher | None = None,
    status: Any | None = None,
) -> SyncPlan:
    """Compute the sync plan by comparing source and destination.

    Returns:
        SyncPlan with all operations to be performed
    """
    filter_matcher = filter_matcher or FilterMatcher()
    is_upload = not _is_bucket_path(source) and _is_bucket_path(dest)
    is_download = _is_bucket_path(source) and not _is_bucket_path(dest)

    if not is_upload and not is_download:
        raise ValueError("One of source or dest must be a bucket path (hf://buckets/...) and the other must be local.")

    plan = SyncPlan(
        source=source,
        dest=dest,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    remote_total: int | None = None
    if is_upload:
        # Local -> Remote
        local_path = os.path.abspath(source)
        parsed = _parse_bucket_uri(dest)
        bucket_id, prefix = parsed.id, parsed.path_in_repo

        if not os.path.isdir(local_path):
            raise ValueError(f"Source must be a directory: {local_path}")

        # Get local and remote file lists
        local_files = {}
        for rel_path, size, mtime_ms in _list_local_files(local_path):
            if filter_matcher.matches(rel_path):
                local_files[rel_path] = (size, mtime_ms)
            if status:
                status.update(f"Scanning local directory ({len(local_files)} files)")
        if status:
            status.done(f"Scanning local directory ({len(local_files)} files)")

        remote_files = {}
        if status:
            try:
                remote_total = api.bucket_info(bucket_id).total_files
            except Exception:
                pass
        try:
            for rel_path, size, mtime_ms, _ in _list_remote_files(api, bucket_id, prefix):
                if filter_matcher.matches(rel_path):
                    remote_files[rel_path] = (size, mtime_ms)
                if status:
                    total_str = f"/{remote_total}" if remote_total is not None else ""
                    status.update(f"Scanning remote bucket ({len(remote_files)}{total_str} files)")
        except BucketNotFoundError:
            # Bucket doesn't exist yet - this is expected for new uploads
            logger.debug(f"Bucket '{bucket_id}' not found, treating as empty.")
        if status:
            status.done(f"Scanning remote bucket ({len(remote_files)} files)")

        # Compare files
        all_paths = set(local_files.keys()) | set(remote_files.keys())
        if status:
            status.done(f"Comparing files ({len(all_paths)} paths)")
        for path in sorted(all_paths):
            local_info = local_files.get(path)
            remote_info = remote_files.get(path)

            if local_info and not remote_info:
                # New file
                if existing:
                    # --existing: skip new files
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=local_info[0],
                            reason="new file (--existing)",
                            local_mtime=_mtime_to_iso(local_info[1]),
                        )
                    )
                else:
                    plan.operations.append(
                        SyncOperation(
                            action="upload",
                            path=path,
                            size=local_info[0],
                            reason="new file",
                            local_mtime=_mtime_to_iso(local_info[1]),
                        )
                    )
            elif local_info and remote_info:
                # File exists in both - use helper to determine action
                local_size, local_mtime = local_info
                remote_size, remote_mtime = remote_info
                plan.operations.append(
                    _compare_files_for_sync(
                        path=path,
                        action="upload",
                        source_size=local_size,
                        source_mtime=local_mtime,
                        dest_size=remote_size,
                        dest_mtime=remote_mtime,
                        source_newer_label="local newer",
                        dest_newer_label="remote newer",
                        ignore_sizes=ignore_sizes,
                        ignore_times=ignore_times,
                        ignore_existing=ignore_existing,
                    )
                )
            elif not local_info and remote_info and delete:
                # File only in remote and --delete mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=remote_info[0],
                        reason="not in source (--delete)",
                        remote_mtime=_mtime_to_iso(remote_info[1]),
                    )
                )

    else:
        # Remote -> Local (download)
        parsed = _parse_bucket_uri(source)
        bucket_id, prefix = parsed.id, parsed.path_in_repo
        local_path = os.path.abspath(dest)

        # Get remote and local file lists
        remote_files = {}
        bucket_file_map: dict[str, Any] = {}
        if status:
            try:
                remote_total = api.bucket_info(bucket_id).total_files
            except Exception:
                pass
        for rel_path, size, mtime_ms, bucket_file in _list_remote_files(api, bucket_id, prefix):
            if filter_matcher.matches(rel_path):
                remote_files[rel_path] = (size, mtime_ms)
                bucket_file_map[rel_path] = bucket_file
            if status:
                total_str = f"/{remote_total}" if remote_total is not None else ""
                status.update(f"Scanning remote bucket ({len(remote_files)}{total_str} files)")
        if status:
            status.done(f"Scanning remote bucket ({len(remote_files)} files)")

        local_files = {}
        if os.path.isdir(local_path):
            if delete:
                # Full walk needed to discover local-only files for deletion.
                for rel_path, size, mtime_ms in _list_local_files(local_path):
                    if filter_matcher.matches(rel_path):
                        local_files[rel_path] = (size, mtime_ms)
                    if status:
                        status.update(f"Scanning local directory ({len(local_files)} files)")
            else:
                # Without --delete, the plan only depends on paths that exist
                # remotely. Stat just those instead of walking the whole tree,
                # which can take minutes when dest sits in a large directory
                # like ~/.cache/huggingface/.
                for rel_path in remote_files:
                    local_file = os.path.join(local_path, rel_path)
                    stat_info = _stat_local(local_file)
                    if stat_info is None:
                        continue
                    local_files[rel_path] = stat_info
                    if status:
                        status.update(f"Scanning local directory ({len(local_files)} files)")
        if status:
            status.done(f"Scanning local directory ({len(local_files)} files)")

        # Compare files
        all_paths = set(remote_files.keys()) | set(local_files.keys())
        if status:
            status.done(f"Comparing files ({len(all_paths)} paths)")
        for path in sorted(all_paths):
            remote_info = remote_files.get(path)
            local_info = local_files.get(path)

            if remote_info and not local_info:
                # New file
                if existing:
                    # --existing: skip new files
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=remote_info[0],
                            reason="new file (--existing)",
                            remote_mtime=_mtime_to_iso(remote_info[1]),
                        )
                    )
                else:
                    plan.operations.append(
                        SyncOperation(
                            action="download",
                            path=path,
                            size=remote_info[0],
                            reason="new file",
                            remote_mtime=_mtime_to_iso(remote_info[1]),
                            bucket_file=bucket_file_map.get(path),
                        )
                    )
            elif remote_info and local_info:
                # File exists in both - use helper to determine action
                remote_size, remote_mtime = remote_info
                local_size, local_mtime = local_info
                plan.operations.append(
                    _compare_files_for_sync(
                        path=path,
                        action="download",
                        source_size=remote_size,
                        source_mtime=remote_mtime,
                        dest_size=local_size,
                        dest_mtime=local_mtime,
                        source_newer_label="remote newer",
                        dest_newer_label="local newer",
                        ignore_sizes=ignore_sizes,
                        ignore_times=ignore_times,
                        ignore_existing=ignore_existing,
                        bucket_file=bucket_file_map.get(path),
                    )
                )
            elif not remote_info and local_info and delete:
                # File only in local and --delete mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=local_info[0],
                        reason="not in source (--delete)",
                        local_mtime=_mtime_to_iso(local_info[1]),
                    )
                )

    return plan


# =============================================================================
# Plan serialization
# =============================================================================


def _write_plan(plan: SyncPlan, f) -> None:
    """Write a sync plan as JSONL to a file-like object."""
    # Write header
    header = {
        "type": "header",
        "source": plan.source,
        "dest": plan.dest,
        "timestamp": plan.timestamp,
        "summary": plan.summary(),
    }
    f.write(json.dumps(header) + "\n")

    # Write operations
    for op in plan.operations:
        op_dict: dict[str, Any] = {
            "type": "operation",
            "action": op.action,
            "path": op.path,
            "reason": op.reason,
        }
        if op.size is not None:
            op_dict["size"] = op.size
        if op.local_mtime is not None:
            op_dict["local_mtime"] = op.local_mtime
        if op.remote_mtime is not None:
            op_dict["remote_mtime"] = op.remote_mtime
        f.write(json.dumps(op_dict) + "\n")


def _save_plan(plan: SyncPlan, plan_file: str) -> None:
    """Save a sync plan to a JSONL file."""
    with open(plan_file, "w") as f:
        _write_plan(plan, f)


def _load_plan(plan_file: str) -> SyncPlan:
    """Load a sync plan from a JSONL file."""
    with open(plan_file) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty plan file: {plan_file}")

    # Parse header
    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError("Invalid plan file: expected header as first line")

    plan = SyncPlan(
        source=header["source"],
        dest=header["dest"],
        timestamp=header["timestamp"],
    )

    # Parse operations
    for line in lines[1:]:
        op_dict = json.loads(line)
        if op_dict.get("type") != "operation":
            continue
        plan.operations.append(
            SyncOperation(
                action=op_dict["action"],
                path=op_dict["path"],
                size=op_dict.get("size"),
                reason=op_dict.get("reason", ""),
                local_mtime=op_dict.get("local_mtime"),
                remote_mtime=op_dict.get("remote_mtime"),
            )
        )

    return plan


# =============================================================================
# Plan execution
# =============================================================================


def _execute_plan(plan: SyncPlan, api: "HfApi", verbose: bool = False, status: Any | None = None) -> None:
    """Execute a sync plan."""
    is_upload = not _is_bucket_path(plan.source) and _is_bucket_path(plan.dest)
    is_download = _is_bucket_path(plan.source) and not _is_bucket_path(plan.dest)

    # Validate here too: an --apply'd plan is read from disk, bypassing the _list_remote_files guard.
    for op in plan.operations:
        _validate_relative_filename(op.path)

    if is_upload:
        local_path = os.path.abspath(plan.source)
        parsed = _parse_bucket_uri(plan.dest)
        bucket_id, prefix = parsed.id, parsed.path_in_repo

        # Collect operations
        add_files: list[tuple[str | Path | bytes, str]] = []
        delete_paths: list[str] = []

        for op in plan.operations:
            match op.action:
                case "upload":
                    local_file = os.path.join(local_path, op.path)
                    remote_path = f"{prefix}/{op.path}" if prefix else op.path
                    if verbose:
                        print(f"  Uploading: {op.path} ({op.reason})")
                    add_files.append((local_file, remote_path))
                case "delete":
                    remote_path = f"{prefix}/{op.path}" if prefix else op.path
                    if verbose:
                        print(f"  Deleting: {op.path} ({op.reason})")
                    delete_paths.append(remote_path)
                case "skip" if verbose:
                    print(f"  Skipping: {op.path} ({op.reason})")

        # Execute batch operations
        if add_files or delete_paths:
            if status:
                parts = []
                if add_files:
                    parts.append(f"uploading {len(add_files)} files")
                if delete_paths:
                    parts.append(f"deleting {len(delete_paths)} files")
                status.done(", ".join(parts).capitalize())
            api.batch_bucket_files(
                bucket_id,
                add=add_files or None,
                delete=delete_paths or None,
            )

    elif is_download:
        parsed = _parse_bucket_uri(plan.source)
        bucket_id, prefix = parsed.id, parsed.path_in_repo
        local_path = os.path.abspath(plan.dest)

        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)

        # Collect download operations
        download_files: list[tuple[str | BucketFile, str | Path]] = []
        delete_files: list[str] = []

        for op in plan.operations:
            if op.action == "download":
                local_file = os.path.join(local_path, op.path)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                if verbose:
                    print(f"  Downloading: {op.path} ({op.reason})")
                # Use BucketFile when available (avoids extra metadata fetch per file)
                if op.bucket_file is not None:
                    download_files.append((op.bucket_file, local_file))
                else:
                    remote_path = f"{prefix}/{op.path}" if prefix else op.path
                    download_files.append((remote_path, local_file))
            elif op.action == "delete":
                local_file = os.path.join(local_path, op.path)
                if verbose:
                    print(f"  Deleting: {op.path} ({op.reason})")
                delete_files.append(local_file)
            elif op.action == "skip" and verbose:
                print(f"  Skipping: {op.path} ({op.reason})")

        # Execute downloads
        if len(download_files) > 0:
            if status:
                status.done(f"Downloading {len(download_files)} files")
            api.download_bucket_files(bucket_id, download_files)

        # Execute deletes
        if status and delete_files:
            status.done(f"Deleting {len(delete_files)} local files")
        for file_path in delete_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                # Remove empty parent directories
                parent = os.path.dirname(file_path)
                while parent != local_path:
                    try:
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    except OSError:
                        break


def _print_plan_summary(plan: SyncPlan) -> None:
    """Print a summary of the sync plan."""
    summary = plan.summary()
    print(f"Sync plan: {plan.source} -> {plan.dest}")
    print(f"  Uploads: {summary['uploads']}")
    print(f"  Downloads: {summary['downloads']}")
    print(f"  Deletes: {summary['deletes']}")
    print(f"  Skips: {summary['skips']}")


# =============================================================================
# Public sync function (Python API)
# =============================================================================


def sync_bucket_internal(
    source: str | None = None,
    dest: str | None = None,
    *,
    api: "HfApi",
    delete: bool = False,
    ignore_times: bool = False,
    ignore_sizes: bool = False,
    existing: bool = False,
    ignore_existing: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    filter_from: str | None = None,
    plan: str | None = None,
    apply: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    token: bool | str | None = None,
) -> SyncPlan:
    """Sync files between a local directory and a bucket.

    This is equivalent to the ``hf buckets sync`` CLI command. One of ``source`` or ``dest`` must be a bucket path
    (``hf://buckets/...``) and the other must be a local directory path.

    Args:
        source (`str`, *optional*):
            Source path: local directory or ``hf://buckets/namespace/bucket_name(/prefix)``.
            Required unless using ``apply``.
        dest (`str`, *optional*):
            Destination path: local directory or ``hf://buckets/namespace/bucket_name(/prefix)``.
            Required unless using ``apply``.
        api ([`HfApi`]):
            The HfApi instance to use for API calls.
        delete (`bool`, *optional*, defaults to `False`):
            Delete destination files not present in source.
        ignore_times (`bool`, *optional*, defaults to `False`):
            Skip files only based on size, ignoring modification times.
        ignore_sizes (`bool`, *optional*, defaults to `False`):
            Skip files only based on modification times, ignoring sizes.
        existing (`bool`, *optional*, defaults to `False`):
            Skip creating new files on receiver (only update existing files).
        ignore_existing (`bool`, *optional*, defaults to `False`):
            Skip updating files that exist on receiver (only create new files).
        include (`list[str]`, *optional*):
            Include files matching patterns (fnmatch-style).
        exclude (`list[str]`, *optional*):
            Exclude files matching patterns (fnmatch-style).
        filter_from (`str`, *optional*):
            Path to a filter file with include/exclude rules.
        plan (`str`, *optional*):
            Save sync plan to this JSONL file instead of executing.
        apply (`str`, *optional*):
            Apply a previously saved plan file. When set, ``source`` and ``dest`` are not needed.
        dry_run (`bool`, *optional*, defaults to `False`):
            Print sync plan to stdout as JSONL without executing.
        verbose (`bool`, *optional*, defaults to `False`):
            Show detailed per-file operations.
        quiet (`bool`, *optional*, defaults to `False`):
            Suppress all output and progress bars.
        token (Union[bool, str, None], optional):
            A valid user access token. If not provided, the locally saved token will be used.

    Returns:
        [`SyncPlan`]: The computed (or loaded) sync plan.

    Raises:
        `ValueError`: If arguments are invalid (e.g., both paths are remote, conflicting options).

    Example:
        ```python
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()

        # Upload local directory to bucket
        >>> api.sync_bucket("./data", "hf://buckets/username/my-bucket")

        # Download bucket to local directory
        >>> api.sync_bucket("hf://buckets/username/my-bucket", "./data")

        # Sync with delete and filtering
        >>> api.sync_bucket(
        ...     "./data",
        ...     "hf://buckets/username/my-bucket",
        ...     delete=True,
        ...     include=["*.safetensors"],
        ... )

        # Dry run: preview what would be synced
        >>> plan = api.sync_bucket("./data", "hf://buckets/username/my-bucket", dry_run=True)
        >>> plan.summary()
        {'uploads': 3, 'downloads': 0, 'deletes': 0, 'skips': 1, 'total_size': 4096}

        # Save plan for review, then apply
        >>> api.sync_bucket("./data", "hf://buckets/username/my-bucket", plan="sync-plan.jsonl")
        >>> api.sync_bucket(apply="sync-plan.jsonl")
        ```
    """
    # Build API with token if needed
    if token is not None:
        from .hf_api import HfApi

        api = HfApi(
            endpoint=api.endpoint,
            token=token,
            library_name=api.library_name,
            library_version=api.library_version,
            user_agent=api.user_agent,
            headers=api.headers,
        )
    # --- Apply mode ---
    if apply:
        if source or dest:
            raise ValueError("Cannot specify source/dest when using apply.")
        if plan is not None:
            raise ValueError("Cannot specify both plan and apply.")
        if delete:
            raise ValueError("Cannot specify delete when using apply.")
        if ignore_times:
            raise ValueError("Cannot specify ignore_times when using apply.")
        if ignore_sizes:
            raise ValueError("Cannot specify ignore_sizes when using apply.")
        if include:
            raise ValueError("Cannot specify include when using apply.")
        if exclude:
            raise ValueError("Cannot specify exclude when using apply.")
        if filter_from:
            raise ValueError("Cannot specify filter_from when using apply.")
        if existing:
            raise ValueError("Cannot specify existing when using apply.")
        if ignore_existing:
            raise ValueError("Cannot specify ignore_existing when using apply.")
        if dry_run:
            raise ValueError("Cannot specify dry_run when using apply.")

        sync_plan = _load_plan(apply)
        status = StatusLine(enabled=not quiet)
        if not quiet:
            _print_plan_summary(sync_plan)
            print("Executing plan...")

        if quiet:
            disable_progress_bars()
        try:
            _execute_plan(sync_plan, api, verbose=verbose, status=status)
        finally:
            if quiet:
                enable_progress_bars()

        if not quiet:
            print("Sync completed.")

        return sync_plan

    # --- Normal mode ---
    if not source or not dest:
        raise ValueError("Both source and dest are required (unless using apply).")

    source_is_bucket = _is_bucket_path(source)
    dest_is_bucket = _is_bucket_path(dest)

    if source_is_bucket and dest_is_bucket:
        raise ValueError("Remote to remote sync is not supported. One path must be local.")

    if not source_is_bucket and not dest_is_bucket:
        raise ValueError("One of source or dest must be a bucket path (hf://buckets/...).")

    if ignore_times and ignore_sizes:
        raise ValueError("Cannot specify both ignore_times and ignore_sizes.")

    if existing and ignore_existing:
        raise ValueError("Cannot specify both existing and ignore_existing.")

    if dry_run and plan:
        raise ValueError("Cannot specify both dry_run and plan.")

    # Validate local path
    if source_is_bucket:
        if os.path.exists(dest) and not os.path.isdir(dest):
            raise ValueError(f"Destination must be a directory: {dest}")
    else:
        if not os.path.isdir(source):
            raise ValueError(f"Source must be an existing directory: {source}")

    # Build filter matcher
    filter_rules = None
    if filter_from:
        filter_rules = _parse_filter_file(filter_from)

    filter_matcher = FilterMatcher(
        include_patterns=include,
        exclude_patterns=exclude,
        filter_rules=filter_rules,
    )

    # Compute sync plan
    status = StatusLine(enabled=not quiet and not dry_run)
    sync_plan = _compute_sync_plan(
        source=source,
        dest=dest,
        api=api,
        delete=delete,
        ignore_times=ignore_times,
        ignore_sizes=ignore_sizes,
        existing=existing,
        ignore_existing=ignore_existing,
        filter_matcher=filter_matcher,
        status=status,
    )

    if dry_run:
        _write_plan(sync_plan, sys.stdout)
        return sync_plan

    if plan:
        _save_plan(sync_plan, plan)
        if not quiet:
            _print_plan_summary(sync_plan)
            print(f"Plan saved to: {plan}")
        return sync_plan

    # Execute plan
    if not quiet:
        _print_plan_summary(sync_plan)

    summary = sync_plan.summary()
    if summary["uploads"] == 0 and summary["downloads"] == 0 and summary["deletes"] == 0:
        if not quiet:
            print("Nothing to sync.")
        return sync_plan

    if not quiet:
        print("Syncing...")

    if quiet:
        disable_progress_bars()
    try:
        _execute_plan(sync_plan, api, verbose=verbose, status=status)
    finally:
        if quiet:
            enable_progress_bars()

    if not quiet:
        print("Sync completed.")

    return sync_plan
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
"""Contains methods to log in to the Hub."""

import html
import os
import subprocess
import sys
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path

from . import constants
from .errors import DeviceCodeError
from .utils import (
    ANSI,
    get_token,
    is_google_colab,
    is_notebook,
    list_credential_helpers,
    logging,
    run_subprocess,
    select_choice,
    set_git_credential,
    tabulate,
    unset_git_credential,
)
from .utils._auth import (
    _get_token_by_name,
    _get_token_from_environment,
    _get_token_from_file,
    _get_token_from_google_colab,
    _read_stored_tokens_full,
    _save_stored_tokens_full,
    _save_token,
    _write_secret,
    get_stored_tokens,
)
from .utils._oauth_device import OAuthTokenResponse, poll_device_token, request_device_code


logger = logging.get_logger(__name__)


def login(
    token: str | None = None,
    *,
    add_to_git_credential: bool = False,
    skip_if_logged_in: bool = True,
) -> None:
    """Login the machine to access the Hub.

    The `token` is persisted in cache and set as a git credential. Once done, the machine
    is logged in and the access token will be available across all `huggingface_hub`
    components. If `token` is not provided, a browser-based OAuth flow is used to
    authenticate: open a URL, enter a short code, and the token is retrieved and saved.
    In a terminal, you can also choose to paste an existing access token instead.

    To log in from outside of a script, one can also use `hf auth login` which is
    a cli command that wraps [`login`].

    > [!TIP]
    > When the token is not passed, [`login`] will automatically detect if the script runs
    > in a notebook or not. However, this detection might not be accurate due to the
    > variety of notebooks that exists nowadays. If that is the case, you can always force
    > the UI by using [`notebook_login`] or [`interpreter_login`].

    Args:
        token (`str`, *optional*):
            User access token to generate from https://huggingface.co/settings/token.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user. Only used when `token`
            is provided; ignored by the browser-based flow.
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If an organization token is passed. Only personal account tokens are valid
            to log in.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If token is invalid.
        [`DeviceCodeError`]
            If the browser-based login fails (authorization denied, code expired, ...).
    """
    if token is not None:
        if not add_to_git_credential:
            logger.info(
                "The token has not been saved to the git credentials helper. Pass "
                "`add_to_git_credential=True` in this function directly or "
                "`--add-to-git-credential` if using via `hf`CLI if "
                "you want to set the git credential as well."
            )
        _validate_and_save_token(token, add_to_git_credential=add_to_git_credential)
        return
    if add_to_git_credential:
        logger.warning(
            "`add_to_git_credential=True` is only supported when a token is passed directly. "
            "It is ignored by the browser-based login."
        )
    if is_notebook():
        notebook_login(skip_if_logged_in=skip_if_logged_in)
    else:
        interpreter_login(skip_if_logged_in=skip_if_logged_in)


def logout(token_name: str | None = None) -> None:
    """Logout the machine from the Hub.

    Token is deleted from the machine and removed from git credential.

    Args:
        token_name (`str`, *optional*):
            Name of the access token to logout from. If `None`, will log out from all saved access tokens.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the access token name is not found.
    """
    if get_token() is None and not get_stored_tokens():  # No active token and no saved access tokens
        logger.warning("Not logged in!")
        return
    if not token_name:
        # Delete all saved access tokens and token
        for file_path in (constants.HF_TOKEN_PATH, constants.HF_STORED_TOKENS_PATH):
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass
        logger.info("Successfully logged out from all access tokens.")
    else:
        _logout_from_token(token_name)
        logger.info(f"Successfully logged out from access token: {token_name}.")

    unset_git_credential()

    # Check if still logged in
    if _get_token_from_google_colab() is not None:
        raise OSError(
            "You are automatically logged in using a Google Colab secret.\n"
            "To log out, you must unset the `HF_TOKEN` secret in your Colab settings."
        )
    if _get_token_from_environment() is not None:
        raise OSError(
            "Token has been deleted from your machine but you are still logged in.\n"
            "To log out, you must clear out both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` environment variables."
        )


def auth_switch(token_name: str, add_to_git_credential: bool = False) -> None:
    """Switch to a different access token.

    Args:
        token_name (`str`):
            Name of the access token to switch to.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user. If `token` is `None`,
            the value of `add_to_git_credential` is ignored and will be prompted again
            to the end user.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the access token name is not found.
    """
    token = _get_token_by_name(token_name)
    if not token:
        raise ValueError(f"Access token {token_name} not found in {constants.HF_STORED_TOKENS_PATH}")
    # Write token to HF_TOKEN_PATH
    _set_active_token(token_name, add_to_git_credential)
    logger.info(f"The current active token is: {token_name}")
    token_from_environment = _get_token_from_environment()
    if token_from_environment is not None and token_from_environment != token:
        logger.warning(
            "The environment variable `HF_TOKEN` is set and will override the access token you've just switched to."
        )


def auth_list() -> None:
    """List all stored access tokens."""
    # Resolve the current token before reading the file: `get_token()` may refresh an OAuth
    # token and rewrite the stored tokens on the way.
    current_token = get_token()
    stored_tokens = _read_stored_tokens_full()

    if not stored_tokens:
        if _get_token_from_environment():
            logger.info("No stored access tokens found.")
            logger.warning("Note: Environment variable `HF_TOKEN` is set and is the current active token.")
        else:
            logger.info("No access tokens found.")
        return
    show_expires = any("expires_at" in fields for fields in stored_tokens.values())
    headers = [" ", "name", "token"] + (["expires"] if show_expires else [])

    current_token_name = None
    rows: list[list[str | int]] = []
    for token_name, fields in stored_tokens.items():
        token = fields.get("hf_token", "<not set>")
        if token == current_token:
            current_token_name = token_name
        masked_token = f"{token[:3]}****{token[-4:]}" if token != "<not set>" else token
        row: list[str | int] = ["*" if token == current_token else "", token_name, masked_token]
        if show_expires:
            row.append(_format_expiration(fields.get("expires_at")))
        rows.append(row)
    print(tabulate(rows, headers=headers))

    if _get_token_from_environment():
        logger.warning(
            "\nNote: Environment variable `HF_TOKEN` is set and is the current active token independently from the stored tokens listed above."
        )
    elif current_token_name is None:
        logger.warning(
            "\nNote: No active token is set and no environment variable `HF_TOKEN` is found. Use `hf auth login` to log in."
        )


###
# Device Code OAuth login (RFC 8628)
###


def _device_code_login() -> None:
    """Run the Device Code OAuth flow: request a code, prompt the user to authorize it in a browser,
    poll for the token and save it."""
    device_info = request_device_code()

    # The complete URI has the code pre-filled when the server supports it.
    print(f"\n    Open this URL in your browser:\n        {device_info['verification_uri_complete']}\n")
    print(f"    And enter the code: {device_info['user_code']}\n")

    print("    Waiting for authorization", end="", flush=True)
    try:
        response = poll_device_token(device_info, on_pending=lambda: print(".", end="", flush=True))
    finally:
        print()  # newline after the progress dots, also on failure

    _save_oauth_token(response)


def _save_oauth_token(response: OAuthTokenResponse) -> tuple[str, str]:
    """Validate and persist a token response from the device code flow, including refresh metadata."""
    expires_in = response.get("expires_in")
    token_name, username = _validate_and_save_token(
        response["access_token"],
        add_to_git_credential=False,
        refresh_token=response.get("refresh_token"),
        expires_at=int(time.time()) + int(expires_in) if expires_in else None,
    )
    if note := _expiration_note(response):
        logger.info(f"Note: {note}")
    return token_name, username


def _expiration_note(response: OAuthTokenResponse) -> str | None:
    """Human-readable note about the lifetime of a freshly obtained OAuth token, if known."""
    expires_in = response.get("expires_in")
    if not expires_in:
        return None
    if response.get("refresh_token"):
        return "This token will be refreshed automatically when it expires."
    days = max(1, int(expires_in) // 86400)
    return f"This token expires in {days} days. Log in again to renew it."


###
# Interpreter-based login (text)
###


def interpreter_login(*, skip_if_logged_in: bool = True) -> None:
    """
    Displays a prompt to log in to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when not run in a notebook.
    [`interpreter_login`] is useful if you want to force the use of the terminal prompt
    instead of a notebook flow.

    For more details, see [`login`].

    Args:
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
    """
    if skip_if_logged_in and get_token() is not None:
        logger.info("User is already logged in. Use `hf auth login --force` to force re-login.")
        return

    if get_token() is not None:
        logger.info("Note: a token is already saved on this machine. Logging in again will replace the active token.")

    if _prompt_login_method() == "token":
        _paste_token_login()
    else:
        _device_code_login()


def _prompt_login_method() -> str:
    """Ask the user how to log in: "browser" (default) or "token". Never prompts without a TTY."""
    if sys.stdin is None or not sys.stdin.isatty():
        return "browser"
    choice = select_choice("How would you like to log in?", ["Log in with your browser", "Paste an access token"])
    return "browser" if choice == 0 else "token"


def _paste_token_login() -> None:
    logger.info(
        "    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens ."
    )
    if os.name == "nt":
        logger.info("Token can be pasted using 'Right-Click'.")
    token = getpass("Enter your token (input will not be visible): ")
    _validate_and_save_token(token=token, add_to_git_credential=False)


###
# Notebook-based login
###


def notebook_login(*, skip_if_logged_in: bool = True) -> None:
    """
    Displays a prompt to log in to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when run in a notebook.
    [`notebook_login`] is useful if you want to force the use of the notebook flow
    instead of a prompt in the terminal.

    For more details, see [`login`].

    Args:
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
    """
    if skip_if_logged_in and get_token() is not None:
        logger.info("User is already logged in. Use `hf auth login --force` to force re-login.")
        return

    try:
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        # Not in a notebook environment: fall back to the terminal flow
        interpreter_login(skip_if_logged_in=False)
        return

    device_info = request_device_code()
    # Escape server-provided values: they end up in raw notebook HTML.
    verification_uri = html.escape(device_info["verification_uri"])
    verification_uri_complete = html.escape(device_info["verification_uri_complete"])

    display(
        HTML(
            '<center><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"'
            ' width="100" alt="Hugging Face"><br><br>'
            "<p>To log in, open this URL and enter the code:</p>"
            f'<p><a href="{verification_uri_complete}" target="_blank"><b>{verification_uri}</b></a></p>'
            '<p style="font-size: 1.6em; letter-spacing: 0.3em; font-family: monospace;">'
            f"<b>{html.escape(device_info['user_code'])}</b></p></center>"
        )
    )
    display(HTML("<center><i>Waiting for authorization...</i></center>"))
    try:
        response = poll_device_token(device_info)
    except DeviceCodeError as e:
        display(HTML(f"<center><b style='color: red;'>Login failed: {html.escape(str(e))}</b></center>"))
        return

    try:
        token_name, username = _save_oauth_token(response)
    except Exception as error:
        display(HTML(f"<center><b style='color: red;'>{html.escape(str(error))}</b></center>"))
        return

    message = f"Login successful. Logged in as <b>{html.escape(username)}</b> (token: <code>{html.escape(token_name)}</code>)."
    if note := _expiration_note(response):
        message += f"<br>{html.escape(note)}"
    display(HTML(f"<center>{message}</center>"))


###
# Login private helpers
###


def _validate_and_save_token(
    token: str,
    add_to_git_credential: bool,
    refresh_token: str | None = None,
    expires_at: int | None = None,
) -> tuple[str, str]:
    """Validate a token against the Hub, save it to the stored tokens file and set it as active.

    The token is stored under its `displayName` from the whoami response, or `oauth-{username}`
    for OAuth tokens (which have no display name).

    Args:
        token (`str`):
            The access token.
        add_to_git_credential (`bool`):
            Whether to save the token to the git credential helpers.
        refresh_token (`str`, *optional*):
            OAuth refresh token to persist alongside the access token.
        expires_at (`int`, *optional*):
            Unix timestamp at which the access token expires.

    Returns:
        `tuple[str, str]`: The token name and the username.
    """
    from .hf_api import whoami  # avoid circular import

    if token.startswith("api_org"):
        raise ValueError("You must use your personal account token, not an organization token.")

    token_info = whoami(token)
    username = token_info["name"]

    access_token_info = (token_info.get("auth") or {}).get("accessToken") or {}
    if role := access_token_info.get("role"):
        logger.info(f"Token is valid (permission: {role}).")
    else:
        logger.info("Token is valid.")

    token_name = access_token_info.get("displayName") or f"oauth-{username}"

    # Store token locally
    _save_token(token=token, token_name=token_name, refresh_token=refresh_token, expires_at=expires_at)
    # Set active token
    _set_active_token(token_name=token_name, add_to_git_credential=add_to_git_credential)
    logger.info("Login successful.")
    if _get_token_from_environment():
        logger.warning(
            "Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured."
        )
    else:
        logger.info(f"The current active token is: `{token_name}`")
    return token_name, username


def _logout_from_token(token_name: str) -> None:
    """Logout from a specific access token.

    Args:
        token_name (`str`):
            The name of the access token to logout from.
    """
    stored_tokens = _read_stored_tokens_full()
    # If there is no access tokens saved or the access token name is not found, do nothing
    if token_name not in stored_tokens:
        return

    fields = stored_tokens.pop(token_name)
    _save_stored_tokens_full(stored_tokens)

    if fields.get("hf_token") == _get_token_from_file():
        logger.warning(f"Active token '{token_name}' has been deleted.")
        Path(constants.HF_TOKEN_PATH).unlink(missing_ok=True)


def _format_expiration(expires_at: str | None) -> str:
    """Format an `expires_at` unix timestamp for display in `auth list`."""
    if not expires_at:
        return ""
    try:
        timestamp = int(expires_at)
    except ValueError:
        return ""
    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    return f"{date_str} (expired)" if timestamp < time.time() else date_str


def _set_active_token(
    token_name: str,
    add_to_git_credential: bool,
) -> None:
    """Set the active access token.

    Args:
        token_name (`str`):
            The name of the token to set as active.
    """
    token = _get_token_by_name(token_name)
    if not token:
        raise ValueError(f"Token {token_name} not found in {constants.HF_STORED_TOKENS_PATH}")
    if add_to_git_credential:
        if _is_git_credential_helper_configured():
            set_git_credential(token)
            logger.info(
                "Your token has been saved in your configured git credential helpers"
                + f" ({','.join(list_credential_helpers())})."
            )
        else:
            logger.warning("Token has not been saved to git credential helper.")
    # Write token to HF_TOKEN_PATH
    _write_secret(Path(constants.HF_TOKEN_PATH), token)
    logger.info(f"Your token has been saved to {constants.HF_TOKEN_PATH}")


def _is_git_credential_helper_configured() -> bool:
    """Check if a git credential helper is configured.

    Warns user if not the case (except for Google Colab where "store" is set by default
    by `huggingface_hub`).
    """
    helpers = list_credential_helpers()
    if len(helpers) > 0:
        return True  # Do not warn: at least 1 helper is set

    # Only in Google Colab to avoid the warning message
    # See https://github.com/huggingface/huggingface_hub/issues/1043#issuecomment-1247010710
    if is_google_colab():
        _set_store_as_git_credential_helper_globally()
        return True  # Do not warn: "store" is used by default in Google Colab

    # Otherwise, warn user
    print(
        ANSI.red(
            "Cannot authenticate through git-credential as no helper is defined on your"
            " machine.\nYou might have to re-authenticate when pushing to the Hugging"
            " Face Hub.\nRun the following command in your terminal in case you want to"
            " set the 'store' credential helper as default.\n\ngit config --global"
            " credential.helper store\n\nRead"
            " https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more"
            " details."
        )
    )
    return False


def _set_store_as_git_credential_helper_globally() -> None:
    """Set globally the credential.helper to `store`.

    To be used only in Google Colab as we assume the user doesn't care about the git
    credential config. It is the only particular case where we don't want to display the
    warning message in [`notebook_login()`].

    Related:
    - https://github.com/huggingface/huggingface_hub/issues/1043
    - https://github.com/huggingface/huggingface_hub/issues/1051
    - https://git-scm.com/docs/git-credential-store
    """
    try:
        run_subprocess("git config --global credential.helper store")
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr)
<|endoftext|>import os
import tempfile
import threading
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Union
from urllib.parse import quote, unquote

import fsspec
import httpx2
from fsspec.callbacks import _DEFAULT_CALLBACK, NoOpCallback, TqdmCallback
from fsspec.config import apply_config
from fsspec.utils import isfilelike

from . import constants
from ._buckets import BucketFile, BucketFolder
from ._commit_api import CommitOperationCopy, CommitOperationDelete
from ._local_folder import _validate_relative_filename
from .errors import (
    BucketNotFoundError,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from .file_download import hf_hub_url, http_get, xet_get
from .hf_api import SPECIAL_REFS_REVISION_REGEX, HfApi, LastCommitInfo, RepoFile, RepoFolder
from .utils import HFValidationError, XetFileData, hf_raise_for_status, http_backoff, http_stream_backoff, parse_hf_uri
from .utils._runtime import is_xet_available
from .utils._xet import XetTokenType, xet_connection_info_refresh_url
from .utils.insecure_hashlib import md5


@dataclass
class HfFileSystemResolvedPath:
    """Top level Data structure containing information about a resolved Hugging Face file system path."""

    root: str
    path: str

    def unresolve(self) -> str:
        return f"{self.root}/{self.path}".rstrip("/")


@dataclass
class HfFileSystemResolvedRepositoryPath(HfFileSystemResolvedPath):
    """Data structure containing information about a resolved path in a repository."""

    repo_type: str
    repo_id: str
    revision: str
    path_in_repo: str
    root: str = field(init=False)
    path: str = field(init=False)
    # The part placed after '@' in the initial path. It can be a quoted or unquoted refs revision.
    # Used to reconstruct the unresolved path to return to the user.
    _raw_revision: str | None = field(default=None, repr=False)

    def __post_init__(self):
        repo_path = constants.REPO_TYPES_URL_PREFIXES.get(self.repo_type, "") + self.repo_id
        if self._raw_revision:
            self.root = f"{repo_path}@{self._raw_revision}"
        elif self.revision != constants.DEFAULT_REVISION:
            self.root = f"{repo_path}@{safe_revision(self.revision)}"
        else:
            self.root = repo_path
        self.path = self.path_in_repo


@dataclass
class HfFileSystemResolvedBucketPath(HfFileSystemResolvedPath):
    """Data structure containing information about a resolved path in a bucket."""

    bucket_id: str
    root: str = field(init=False)

    def __post_init__(self):
        self.root = "buckets/" + self.bucket_id


# We need to improve fsspec.spec._Cached which is AbstractFileSystem's metaclass
_cached_base: Any = type(fsspec.AbstractFileSystem)


class _Cached(_cached_base):
    """
    Metaclass for caching HfFileSystem instances according to the args.

    This creates an additional reference to the filesystem, which prevents the
    filesystem from being garbage collected when all *user* references go away.
    A call to the :meth:`AbstractFileSystem.clear_instance_cache` must *also*
    be made for a filesystem instance to be garbage collected.

    This is a slightly modified version of `fsspec.spec._Cached` to improve it.
    In particular in `_tokenize` the pid isn't taken into account for the
    `fs_token` used to identify cached instances. The `fs_token` logic is also
    robust to defaults values and the order of the args. Finally new instances
    reuse the states from sister instances in the main thread.
    """

    def __init__(cls, *args, **kwargs):
        # Hack: override https://github.com/fsspec/filesystem_spec/blob/dcb167e8f50e6273d4cfdfc4cab8fc5aa4c958bf/fsspec/spec.py#L53
        super().__init__(*args, **kwargs)
        # Note: we intentionally create a reference here, to avoid garbage
        # collecting instances when all other references are gone. To really
        # delete a FileSystem, the cache must be cleared.
        cls._cache = {}

    def __call__(cls, *args, **kwargs):
        # Hack: override https://github.com/fsspec/filesystem_spec/blob/dcb167e8f50e6273d4cfdfc4cab8fc5aa4c958bf/fsspec/spec.py#L65
        # Apply fsspec config (env vars / config files) before tokenizing so that
        # HfFileSystem picks up defaults the same way other fsspec filesystems do.
        kwargs = apply_config(cls, kwargs)
        skip = kwargs.pop("skip_instance_cache", False)
        fs_token = cls._tokenize(cls, threading.get_ident(), *args, **kwargs)
        fs_token_main_thread = cls._tokenize(cls, threading.main_thread().ident, *args, **kwargs)
        if not skip and cls.cachable and fs_token in cls._cache:
            # reuse cached instance
            cls._latest = fs_token
            return cls._cache[fs_token]
        else:
            # create new instance
            obj = type.__call__(cls, *args, **kwargs)
            if not skip and cls.cachable and fs_token_main_thread in cls._cache:
                # reuse the cache from the main thread instance in the new instance
                instance_state = cls._cache[fs_token_main_thread]._get_instance_state()
                for attr, state_value in instance_state.items():
                    setattr(obj, attr, state_value)
            obj._fs_token_ = fs_token
            obj.storage_args = args
            obj.storage_options = kwargs
            if cls.cachable and not skip:
                cls._latest = fs_token
                cls._cache[fs_token] = obj
            return obj


class HfFileSystem(fsspec.AbstractFileSystem, metaclass=_Cached):  # ty: ignore[conflicting-metaclass]
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    > [!WARNING]
    > [`HfFileSystem`] provides fsspec compatibility, which is useful for libraries that require it (e.g., reading
    >     Hugging Face datasets directly with `pandas`). However, it introduces additional overhead due to this compatibility
    >     layer. For better performance and reliability, it's recommended to use `HfApi` methods when possible.

    The file system supports paths for the `hf://` protocol, which follows those URL schemes:

    * Models, Datasets and Spaces repositories:

        ```
        hf://<repo-id>[@<revision>]/<path/in/repo>
        hf://datasets/<repo-id>[@<revision>]/<path/in/repo>
        hf://spaces/<repo-id>[@<revision>]/<path/in/repo>
        ```

    * Buckets (generic storage):

        ```
        hf://buckets/<bucket-id>/<path/in/bucket>
        ```

    Note: when using the [`HfFileSystem`] directly, passing the `hf://` protocol prefix is optional in paths.

    Args:
        endpoint (`str`, *optional*):
                Endpoint of the Hub. Defaults to <https://huggingface.co>.
        token (`bool` or `str`, *optional*):
            A valid user access token (string). Defaults to the locally saved
            token, which is the recommended method for authentication (see
            https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
            To disable authentication, pass `False`.
        block_size (`int`, *optional*):
            Block size for reading and writing files.
        expand_info (`bool`, *optional*):
            Whether to expand the information of the files.
        **storage_options (`dict`, *optional*):
            Additional options for the filesystem. See [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.__init__).

    Usage:

    ```python
    >>> from huggingface_hub import hffs

    >>> # List files
    >>> hffs.glob("my-username/my-model/*.bin")
    ['my-username/my-model/pytorch_model.bin']
    >>> hffs.ls("datasets/my-username/my-dataset", detail=False)
    ['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

    >>> # Read/write files
    >>> with hffs.open("my-username/my-model/pytorch_model.bin") as f:
    ...     data = f.read()
    >>> with hffs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
    ...     f.write(data)
    ```

    Specify a token for authentication:
    ```python
    >>> from huggingface_hub import HfFileSystem
    >>> hffs = HfFileSystem(token=token)
    ```
    """

    root_marker = ""
    protocol = "hf"

    def __init__(
        self,
        *args,
        endpoint: str | None = None,
        token: bool | str | None = None,
        block_size: int | None = None,
        expand_info: bool | None = None,
        **storage_options,
    ):
        super().__init__(*args, **storage_options)
        self.endpoint = endpoint or constants.ENDPOINT
        self.token = token
        self._api = HfApi(endpoint=endpoint, token=token)
        self.block_size = block_size
        self.expand_info = expand_info
        # Maps (repo_type, repo_id, revision) to a 2-tuple with:
        #  * the 1st element indicating whether the repository and the revision exist
        #  * the 2nd element being the exception raised if the repository or revision doesn't exist
        self._repo_and_revision_exists_cache: dict[tuple[str, str, str | None], tuple[bool, Exception | None]] = {}
        # Same for buckets
        self._bucket_exists_cache: dict[str, tuple[bool, Exception | None]] = {}
        # Note: special case for buckets: revision is always None
        # Maps parent directory path to path infos
        self.dircache: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def _tokenize(cls, threading_ident: int, *args, **kwargs) -> str:
        """Deterministic token for caching"""
        # make fs_token robust to default values and to kwargs order
        kwargs["endpoint"] = kwargs.get("endpoint") or constants.ENDPOINT
        kwargs["token"] = kwargs.get("token")
        kwargs = {key: kwargs[key] for key in sorted(kwargs)}
        # contrary to fsspec, we don't include pid here
        tokenize_args = (cls, threading_ident, args, kwargs)
        h = md5(str(tokenize_args).encode())
        return h.hexdigest()

    def _repo_and_revision_exist(
        self, repo_type: str, repo_id: str, revision: str | None
    ) -> tuple[bool, Exception | None]:
        if (repo_type, repo_id, revision) not in self._repo_and_revision_exists_cache:
            try:
                self._api.repo_info(
                    repo_id, revision=revision, repo_type=repo_type, timeout=constants.HF_HUB_ETAG_TIMEOUT
                )
            except (RepositoryNotFoundError, HFValidationError) as e:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = False, e
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = False, e
            except RevisionNotFoundError as e:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = False, e
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, None
            else:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = True, None
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, None
        return self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)]

    def _bucket_exists(self, bucket_id: str) -> tuple[bool, Exception | None]:
        if bucket_id not in self._bucket_exists_cache:
            try:
                self._api.bucket_info(bucket_id)
            except BucketNotFoundError as e:
                self._bucket_exists_cache[bucket_id] = False, e
            else:
                self._bucket_exists_cache[bucket_id] = True, None
        return self._bucket_exists_cache[bucket_id]

    def resolve_path(
        self, path: str, revision: str | None = None
    ) -> HfFileSystemResolvedRepositoryPath | HfFileSystemResolvedBucketPath:
        """
        Resolve a Hugging Face file system path into its components.

        Args:
            path (`str`):
                Path to resolve.
            revision (`str`, *optional*):
                The revision of the repo to resolve. Defaults to the revision specified in the path.

        Returns:
            [`HfFileSystemResolvedPath`]: Resolved path information containing `repo_type`, `repo_id`, `revision` and `path_in_repo`.

        Raises:
            `ValueError`:
                If path contains conflicting revision information.
            `NotImplementedError`:
                If trying to list repositories.
        """
        path = self._strip_protocol(path)
        if not path:
            raise NotImplementedError("Access to buckets and repositories lists is not implemented.")
        if path.count("/") == 0:
            raise ValueError(
                f"Repository id must be 'namespace/name', got '{path}'. Single-segment ids (e.g. 'gpt2') are no longer supported."
            )

        parsed = parse_hf_uri(f"{constants.HF_PROTOCOL}{path}")

        # --- Buckets ---
        if parsed.is_bucket:
            bucket_exists, err = self._bucket_exists(parsed.id)
            if not bucket_exists:
                _raise_file_not_found(path, err)
            return HfFileSystemResolvedBucketPath(bucket_id=parsed.id, path=parsed.path_in_repo)

        # --- Repositories ---
        # Align revision from path with explicit revision argument
        if revision is not None and parsed.revision is not None and parsed.revision != revision:
            # The caller provided an explicit revision that conflicts with what parse_hf_uri
            # parsed. This can happen when a user has a branch literally named "refs" and a
            # file at "pr/10" — parse_hf_uri would greedily match "refs/pr/10" as a special
            # ref. Fall back to simple '@' splitting so the caller's revision wins.
            path_without_type = path.split("/", 1)[1] if path.split("/")[0] in constants.HF_URI_TYPE_PREFIXES else path
            repo_id, after_at = path_without_type.split("@", 1)
            revision_in_path, path_in_repo = after_at.split("/", 1) if "/" in after_at else (after_at, "")
            revision_in_path_decoded = unquote(revision_in_path)
            if revision_in_path_decoded != revision:
                raise ValueError(
                    f'Revision specified in path ("{revision_in_path_decoded}") and in `revision` argument ("{revision}") are not the same.'
                )
            repo_and_revision_exist, err = self._repo_and_revision_exist(parsed.type, repo_id, revision)
            if not repo_and_revision_exist:
                _raise_file_not_found(path, err)
            return HfFileSystemResolvedRepositoryPath(
                parsed.type, repo_id, revision, path_in_repo, _raw_revision=revision_in_path
            )

        if parsed.revision is not None and revision is None:
            revision = parsed.revision

        repo_and_revision_exist, err = self._repo_and_revision_exist(parsed.type, parsed.id, revision)
        if not repo_and_revision_exist:
            _raise_file_not_found(path, err)

        # Extract raw revision from original path for unresolve() fidelity
        raw_revision: str | None = None
        if "@" in path and parsed.revision is not None:
            path_without_type = path.split("/", 1)[1] if path.split("/")[0] in constants.HF_URI_TYPE_PREFIXES else path
            raw_after_at = path_without_type.split("@", 1)[1]
            raw_revision = raw_after_at[: -(len(parsed.path_in_repo) + 1)] if parsed.path_in_repo else raw_after_at

        revision = revision if revision is not None else constants.DEFAULT_REVISION
        return HfFileSystemResolvedRepositoryPath(
            parsed.type, parsed.id, revision, parsed.path_in_repo, _raw_revision=raw_revision
        )

    def invalidate_cache(self, path: str | None = None) -> None:
        """
        Clear the cache for a given path.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.invalidate_cache).

        Args:
            path (`str`, *optional*):
                Path to clear from cache. If not provided, clear the entire cache.

        """
        if not path:
            self.dircache.clear()
            self._repo_and_revision_exists_cache.clear()
            self._bucket_exists_cache.clear()
        else:
            try:
                resolved_path = self.resolve_path(path)
            except FileNotFoundError:
                # repo, revision or bucket cached as missing: forget misses so the next lookup hits the Hub
                self._repo_and_revision_exists_cache = {
                    k: v for k, v in self._repo_and_revision_exists_cache.items() if v[0]
                }
                self._bucket_exists_cache = {k: v for k, v in self._bucket_exists_cache.items() if v[0]}
                return
            path = resolved_path.unresolve()
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

            # Only clear repo cache if path is to repo root
            if not resolved_path.path:
                if isinstance(resolved_path, HfFileSystemResolvedRepositoryPath):
                    self._repo_and_revision_exists_cache.pop(
                        (resolved_path.repo_type, resolved_path.repo_id, None), None
                    )
                    self._repo_and_revision_exists_cache.pop(
                        (resolved_path.repo_type, resolved_path.repo_id, resolved_path.revision), None
                    )
                else:
                    self._bucket_exists_cache.pop(resolved_path.bucket_id, None)

    def _open(  # type: ignore
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> Union["HfFileSystemFile", "HfFileSystemStreamFile"]:
        block_size = block_size if block_size is not None else self.block_size
        if block_size is not None:
            kwargs["block_size"] = block_size
        if "a" in mode:
            raise NotImplementedError("Appending to remote files is not yet supported.")
        if block_size == 0:
            return HfFileSystemStreamFile(self, path, mode=mode, revision=revision, **kwargs)
        else:
            return HfFileSystemFile(self, path, mode=mode, revision=revision, **kwargs)

    def _rm(self, path: str, revision: str | None = None, **kwargs) -> None:
        resolved_path = self.resolve_path(path, revision=revision)
        if isinstance(resolved_path, HfFileSystemResolvedBucketPath):
            self._api.batch_bucket_files(resolved_path.bucket_id, delete=[resolved_path.path])
        else:
            self._api.delete_file(
                path_in_repo=resolved_path.path_in_repo,
                repo_id=resolved_path.repo_id,
                token=self.token,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
                commit_message=kwargs.get("commit_message"),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path=resolved_path.unresolve())

    def rm(
        self,
        path: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        """
        Delete files from a repository.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm).

        > [!WARNING]
        > Note: When possible, use `HfApi.delete_file()` for better performance.

        Args:
            path (`str`):
                Path to delete.
            recursive (`bool`, *optional*):
                If True, delete directory and all its contents. Defaults to False.
            maxdepth (`int`, *optional*):
                Maximum number of subdirectories to visit when deleting recursively.
            revision (`str`, *optional*):
                The git revision to delete from.

        """
        resolved_path = self.resolve_path(path, revision=revision)
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth, revision=revision)
        if isinstance(resolved_path, HfFileSystemResolvedBucketPath):
            delete = [self.resolve_path(path).path for path in paths if not self.isdir(path)]
            self._api.batch_bucket_files(resolved_path.bucket_id, delete=delete)
        else:
            paths_in_repo = [self.resolve_path(path).path for path in paths if not self.isdir(path)]
            operations = [CommitOperationDelete(path_in_repo=path_in_repo) for path_in_repo in paths_in_repo]
            commit_message = f"Delete {path} "
            commit_message += "recursively " if recursive else ""
            commit_message += f"up to depth {maxdepth} " if maxdepth is not None else ""
            # TODO: use `commit_description` to list all the deleted paths?
            self._api.create_commit(
                repo_id=resolved_path.repo_id,
                repo_type=resolved_path.repo_type,
                token=self.token,
                operations=operations,
                revision=resolved_path.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path=resolved_path.unresolve())

    def ls(
        self, path: str, detail: bool = True, refresh: bool = False, revision: str | None = None, **kwargs
    ) -> list[str | dict[str, Any]]:
        """
        List the contents of a directory.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.ls).

        > [!WARNING]
        > Note: When possible, use `HfApi.list_repo_tree()` for better performance.

        Args:
            path (`str`):
                Path to the directory.
            detail (`bool`, *optional*):
                If True, returns a list of dictionaries containing file information. If False,
                returns a list of file paths. Defaults to True.
            refresh (`bool`, *optional*):
                If True, bypass the cache and fetch the latest data. Defaults to False.
            revision (`str`, *optional*):
                The git revision to list from.

        Returns:
            `list[Union[str, dict[str, Any]]]`: List of file paths (if detail=False) or list of file information
            dictionaries (if detail=True).
        """
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        try:
            out = self._ls_tree(path, refresh=refresh, revision=revision, **kwargs)
        except EntryNotFoundError:
            # Path could be a file
            if not resolved_path.path:
                _raise_file_not_found(path, None)
            try:
                out = self._ls_tree(self._parent(path), refresh=refresh, revision=revision, **kwargs)
            except EntryNotFoundError:
                out = []
            out = [o for o in out if o["name"] == path]
            if len(out) == 0:
                _raise_file_not_found(path, None)
        return out if detail else [o["name"] for o in out]

    def _ls_tree(
        self,
        path: str,
        recursive: bool = False,
        refresh: bool = False,
        revision: str | None = None,
        expand_info: bool | None = None,
        maxdepth: int | None = None,
    ):
        expand_info = (
            expand_info if expand_info is not None else (self.expand_info if self.expand_info is not None else False)
        )
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        root_path = resolved_path.root
        maxdepth = maxdepth if recursive else 1

        out = []
        if path in self.dircache and not refresh:
            cached_path_infos = self.dircache[path]
            out.extend(cached_path_infos)
            dirs_not_in_dircache = []
            if recursive:
                # Use BFS to traverse the cache and build the "recursive "output
                # (The Hub uses a so-called "tree first" strategy for the tree endpoint but we sort the output to follow the spec so the result is (eventually) the same)
                depth = 2
                dirs_to_visit = deque(
                    [(depth, path_info) for path_info in cached_path_infos if path_info["type"] == "directory"]
                )
                while dirs_to_visit:
                    depth, dir_info = dirs_to_visit.popleft()
                    if maxdepth is None or depth <= maxdepth:
                        if dir_info["name"] not in self.dircache:
                            dirs_not_in_dircache.append(dir_info["name"])
                        else:
                            cached_path_infos = self.dircache[dir_info["name"]]
                            out.extend(cached_path_infos)
                            dirs_to_visit.extend(
                                [
                                    (depth + 1, path_info)
                                    for path_info in cached_path_infos
                                    if path_info["type"] == "directory"
                                ]
                            )

            dirs_not_expanded = []
            if expand_info and isinstance(resolved_path, HfFileSystemResolvedRepositoryPath):
                # Check if there are directories in repos with non-expanded entries
                dirs_not_expanded = [self._parent(o["name"]) for o in out if o["last_commit"] is None]

            if (recursive and dirs_not_in_dircache) or (expand_info and dirs_not_expanded):
                # If the dircache is incomplete, find the common path of the missing and non-expanded entries
                # and extend the output with the result of `_ls_tree(common_path, recursive=True)`
                common_prefix = os.path.commonprefix(dirs_not_in_dircache + dirs_not_expanded)
                # Get the parent directory if the common prefix itself is not a directory
                common_path = (
                    common_prefix.rstrip("/")
                    if common_prefix.endswith("/")
                    or common_prefix == root_path
                    or common_prefix in chain(dirs_not_in_dircache, dirs_not_expanded)
                    else self._parent(common_prefix)
                )
                if maxdepth is not None:
                    common_path_depth = common_path[len(path) :].count("/")
                    maxdepth -= common_path_depth
                out = [o for o in out if not o["name"].startswith(common_path + "/")]
                for cached_path in list(self.dircache):
                    if cached_path.startswith(common_path + "/"):
                        self.dircache.pop(cached_path, None)
                self.dircache.pop(common_path, None)
                out.extend(
                    self._ls_tree(
                        common_path,
                        recursive=recursive,
                        refresh=True,
                        revision=revision,
                        expand_info=expand_info,
                        maxdepth=maxdepth,
                    )
                )
        else:
            tree: Iterable[RepoFile | RepoFolder | BucketFile | BucketFolder]
            if isinstance(resolved_path, HfFileSystemResolvedBucketPath):
                tree = self._list_bucket_tree_with_folders(
                    resolved_path.bucket_id,
                    prefix=resolved_path.path,
                    recursive=recursive,
                )
            else:
                tree = self._api.list_repo_tree(
                    resolved_path.repo_id,
                    resolved_path.path,
                    recursive=recursive,
                    expand=expand_info,
                    revision=resolved_path.revision,
                    repo_type=resolved_path.repo_type,
                )
            listings: dict[str, list[dict[str, Any]]] = {}
            for path_info in tree:
                cache_path = root_path + "/" + path_info.path
                if isinstance(path_info, RepoFile):
                    cache_path_info = {
                        "name": cache_path,
                        "size": path_info.size,
                        "type": "file",
                        "blob_id": path_info.blob_id,
                        "lfs": path_info.lfs,
                        "xet_hash": path_info.xet_hash,
                        "last_commit": path_info.last_commit,
                        "security": path_info.security,
                    }
                elif isinstance(path_info, BucketFile):
                    cache_path_info = {
                        "name": cache_path,
                        "size": path_info.size,
                        "type": "file",
                        "xet_hash": path_info.xet_hash,
                        "mtime": path_info.mtime,
                        "uploaded_at": path_info.uploaded_at,
                    }
                elif isinstance(path_info, RepoFolder):
                    cache_path_info = {
                        "name": cache_path,
                        "size": 0,
                        "type": "directory",
                        "tree_id": path_info.tree_id,
                        "last_commit": path_info.last_commit,
                    }
                else:
                    cache_path_info = {
                        "name": cache_path,
                        "size": 0,
                        "type": "directory",
                        "uploaded_at": path_info.uploaded_at,
                    }
                if cache_path != path:  # listing a bucket file returns the file, not its folder's listing
                    listings.setdefault(self._parent(cache_path), []).append(cache_path_info)
                depth = cache_path[len(path) :].count("/")
                if maxdepth is None or depth <= maxdepth:
                    out.append(cache_path_info)
            # replace cached listings, only once the listing is complete
            self.dircache.update(listings)
            if not out:
                # Empty bucket: `info()` on a child path reads `dircache[parent]`
                self.dircache[path] = []
        return out

    def _list_bucket_tree_with_folders(
        self, bucket_id: str, prefix: str, recursive: bool
    ) -> Iterable[BucketFile | BucketFolder]:
        """Same as `HfApi.list_bucket_tree` but always includes folders"""
        bucket_files = self._api.list_bucket_tree(bucket_id, prefix, recursive=recursive)
        bucket_folders: dict[str, BucketFolder] = {}
        min_depth = 1 + prefix.count("/") if prefix else 0
        out: list[BucketFile | BucketFolder] = []

        for bucket_entry in bucket_files:
            # The server matches `prefix` lexically, so listing "logs" also returns "logs_existing/...".
            # Filesystem semantics require path components => drop entries that aren't `prefix` itself
            # or below it.
            if prefix and bucket_entry.path != prefix and not bucket_entry.path.startswith(f"{prefix}/"):
                continue
            out.append(bucket_entry)

            # If recursive=False, both files and folders are returned by the server => nothing to do
            if not recursive:
                continue

            # Otherwise, let's rebuild BucketFolders manually
            for parent_bucket_folder_str in list(PurePosixPath(bucket_entry.path).parents)[: -min_depth - 1]:
                parent_bucket_folder = BucketFolder(
                    type="directory", path=str(parent_bucket_folder_str), uploaded_at=bucket_entry.uploaded_at
                )

                # If folder not visited yet, add it
                if parent_bucket_folder.path not in bucket_folders:
                    out.append(parent_bucket_folder)
                    bucket_folders[parent_bucket_folder.path] = parent_bucket_folder
                    continue

                # Otherwise, get back BucketFolder object and update its 'uploaded_at'
                if parent_bucket_folder.uploaded_at is not None:
                    bucket_folder = bucket_folders[parent_bucket_folder.path]
                    if bucket_folder.uploaded_at is None or (
                        bucket_folder.uploaded_at < parent_bucket_folder.uploaded_at
                    ):
                        bucket_folder.uploaded_at = parent_bucket_folder.uploaded_at

        # An empty listing is only a "not found" when a specific prefix was requested. Listing the
        # bucket root itself is valid even with zero objects (e.g. freshly created bucket).
        if prefix and not out:
            raise EntryNotFoundError(f"File not found in bucket '{bucket_id}': '{prefix}'")
        return out

    def walk(self, path: str, *args, **kwargs) -> Iterator[tuple[str, list[str], list[str]]]:
        """
        Return all files below the given path.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.walk).

        Args:
            path (`str`):
                Root path to list files from.

        Returns:
            `Iterator[tuple[str, list[str], list[str]]]`: An iterator of (path, list of directory names, list of file names) tuples.
        """
        path = self.resolve_path(path, revision=kwargs.get("revision")).unresolve()
        yield from super().walk(path, *args, **kwargs)

    def glob(self, path: str, maxdepth: int | None = None, **kwargs) -> list[str]:
        """
        Find files by glob-matching.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.glob).

        Args:
            path (`str`):
                Path pattern to match.
            maxdepth (`int`, *optional*):
                Maximum depth to descend into directories. By default, no limit.

        Returns:
            `list[str]`: List of paths matching the pattern.
        """
        path = self.resolve_path(path, revision=kwargs.get("revision")).unresolve()
        return super().glob(path, maxdepth=maxdepth, **kwargs)

    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        refresh: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> list[str] | dict[str, dict[str, Any]]:
        """
        List all files below path.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.find).

        Args:
            path (`str`):
                Root path to list files from.
            maxdepth (`int`, *optional*):
                Maximum depth to descend into subdirectories.
            withdirs (`bool`, *optional*):
                Include directory paths in the output. Defaults to False.
            detail (`bool`, *optional*):
                If True, returns a dict mapping paths to file information. Defaults to False.
            refresh (`bool`, *optional*):
                If True, bypass the cache and fetch the latest data. Defaults to False.
            revision (`str`, *optional*):
                The git revision to list from.

        Returns:
            `Union[list[str], dict[str, dict[str, Any]]]`: List of paths or dict of file information.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        try:
            out = self._ls_tree(path, recursive=True, refresh=refresh, maxdepth=maxdepth, **kwargs)
        except EntryNotFoundError:
            # Path could be a file
            try:
                if self.info(path, revision=revision, **kwargs)["type"] == "file":
                    out = {path: {}}
                else:
                    out = {}
            except FileNotFoundError:
                out = {}
        else:
            if not withdirs:
                out = [o for o in out if o["type"] != "directory"]
            else:
                # If `withdirs=True`, include the directory itself to be consistent with the spec
                path_info = self.info(path, **kwargs)
                out = [path_info] + out if path_info["type"] == "directory" else out
            out = {o["name"]: o for o in out}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    def cp_file(self, path1: str, path2: str, revision: str | None = None, **kwargs) -> None:
        """
        Copy a file within or between repositories.

        > [!WARNING]
        > Note: When possible, use `HfApi.upload_file()` for better performance.

        Args:
            path1 (`str`):
                Source path to copy from.
            path2 (`str`):
                Destination path to copy to.
            revision (`str`, *optional*):
                The git revision to copy from.

        """
        resolved_path1 = self.resolve_path(path1, revision=revision)
        resolved_path2 = self.resolve_path(path2, revision=revision)
        if isinstance(resolved_path1, HfFileSystemResolvedBucketPath) or isinstance(
            resolved_path2, HfFileSystemResolvedBucketPath
        ):
            raise NotImplementedError("Copy from/to buckets is not available yet")

        same_repo = (
            resolved_path1.repo_type == resolved_path2.repo_type and resolved_path1.repo_id == resolved_path2.repo_id
        )

        if same_repo:
            commit_message = f"Copy {path1} to {path2}"
            self._api.create_commit(
                repo_id=resolved_path1.repo_id,
                repo_type=resolved_path1.repo_type,
                revision=resolved_path2.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description", ""),
                operations=[
                    CommitOperationCopy(
                        src_path_in_repo=resolved_path1.path_in_repo,
                        path_in_repo=resolved_path2.path_in_repo,
                        src_revision=resolved_path1.revision,
                    )
                ],
            )
        else:
            with self.open(path1, "rb", revision=resolved_path1.revision) as f:
                content = f.read()
            commit_message = f"Copy {path1} to {path2}"
            self._api.upload_file(
                path_or_fileobj=content,
                path_in_repo=resolved_path2.path_in_repo,
                repo_id=resolved_path2.repo_id,
                token=self.token,
                repo_type=resolved_path2.repo_type,
                revision=resolved_path2.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path=resolved_path1.unresolve())
        self.invalidate_cache(path=resolved_path2.unresolve())

    def modified(self, path: str, **kwargs) -> datetime:
        """
        Get the last modified time of a file.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.modified).

        Args:
            path (`str`):
                Path to the file.

        Returns:
            `datetime`: Last modified time of the file.
        """
        info = self.info(path, **{**kwargs, "expand_info": True})  # type: ignore
        if "last_commit" in info:
            if info["last_commit"] is None:
                raise NotImplementedError(f"'modified' is not implemented for repository paths like '{path}'")
            return info["last_commit"].date
        elif "mtime" in info and info["mtime"]:
            return info["mtime"]
        elif "uploaded_at" in info and info["uploaded_at"]:
            return info["uploaded_at"]
        else:
            raise NotImplementedError(f"Cannot determined 'modified' for path '{path}' (info: {info})")

    def info(self, path: str, refresh: bool = False, revision: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Get information about a file or directory.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.info).

        > [!WARNING]
        > Note: When possible, use `HfApi.get_paths_info()` or `HfApi.repo_info()`  for better performance
        > (or `HfApi.get_bucket_paths_info()` or `HfApi.bucket_info()` for buckets)

        Args:
            path (`str`):
                Path to get info for.
            refresh (`bool`, *optional*):
                If True, bypass the cache and fetch the latest data. Defaults to False.
            revision (`str`, *optional*):
                The git revision to get info from.

        Returns:
            `dict[str, Any]`: Dictionary containing file information (type, size, commit info, etc.).

        """
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        expand_info = kwargs.get(
            "expand_info", self.expand_info if self.expand_info is not None else False
        )  # don't expose it as a parameter in the public API to follow the spec
        out: dict[str, Any] | None
        if not resolved_path.path:
            # Path is the root directory
            out = {
                "name": path,
                "size": 0,
                "type": "directory",
            }
            if isinstance(resolved_path, HfFileSystemResolvedRepositoryPath):
                out["last_commit"] = None
            if isinstance(resolved_path, HfFileSystemResolvedRepositoryPath) and expand_info:
                last_commit = self._api.list_repo_commits(
                    resolved_path.repo_id, repo_type=resolved_path.repo_type, revision=resolved_path.revision
                )[-1]
                out = {
                    **out,
                    "tree_id": None,  # TODO: tree_id of the root directory?
                    "last_commit": LastCommitInfo(
                        oid=last_commit.commit_id, title=last_commit.title, date=last_commit.created_at
                    ),
                }
        elif isinstance(resolved_path, HfFileSystemResolvedBucketPath):
            parent_path = self._parent(path)
            # Fill the cache with cheap call
            self.ls(parent_path, refresh=refresh)
            out1 = [o for o in self.dircache[parent_path] if o["name"] == path]
            if not out1:
                _raise_file_not_found(path, None)
            out = out1[0]
        else:
            out = None
            parent_path = self._parent(path)
            if not expand_info and parent_path not in self.dircache:
                # Fill the cache with cheap call
                self.ls(parent_path)
            if parent_path in self.dircache:
                # Check if the path is in the cache
                out1 = [o for o in self.dircache[parent_path] if o["name"] == path]
                if not out1:
                    _raise_file_not_found(path, None)
                out = out1[0]
            if refresh or out is None or (expand_info and out and out["last_commit"] is None):
                paths_info = self._api.get_paths_info(
                    resolved_path.repo_id,
                    resolved_path.path_in_repo,
                    expand=expand_info,
                    revision=resolved_path.revision,
                    repo_type=resolved_path.repo_type,
                )
                if not paths_info:
                    _raise_file_not_found(path, None)
                path_info = paths_info[0]
                root_path = HfFileSystemResolvedRepositoryPath(
                    resolved_path.repo_type,
                    resolved_path.repo_id,
                    resolved_path.revision,
                    path_in_repo="",
                    _raw_revision=resolved_path._raw_revision,
                ).unresolve()
                if isinstance(path_info, RepoFile):
                    out = {
                        "name": root_path + "/" + path_info.path,
                        "size": path_info.size,
                        "type": "file",
                        "blob_id": path_info.blob_id,
                        "lfs": path_info.lfs,
                        "xet_hash": path_info.xet_hash,
                        "last_commit": path_info.last_commit,
                        "security": path_info.security,
                    }
                else:
                    out = {
                        "name": root_path + "/" + path_info.path,
                        "size": 0,
                        "type": "directory",
                        "tree_id": path_info.tree_id,
                        "last_commit": path_info.last_commit,
                    }
                if not expand_info:
                    out = {k: out[k] for k in ["name", "size", "type"]}
        assert out is not None
        return out

    def exists(self, path, **kwargs):
        """
        Check if a file exists.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.exists).

        > [!WARNING]
        > Note: When possible, use `HfApi.file_exists()` for better performance.

        Args:
            path (`str`):
                Path to check.

        Returns:
            `bool`: True if file exists, False otherwise.
        """
        try:
            if kwargs.get("refresh", False):
                self.invalidate_cache(path)

            self.info(path, **kwargs)
            return True
        except OSError:
            return False

    def isdir(self, path):
        """
        Check if a path is a directory.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isdir).

        Args:
            path (`str`):
                Path to check.

        Returns:
            `bool`: True if path is a directory, False otherwise.
        """
        try:
            return self.info(path)["type"] == "directory"
        except OSError:
            return False

    def isfile(self, path):
        """
        Check if a path is a file.

        For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isfile).

        Args:
            path (`str`):
                Path to check.

        Returns:
            `bool`: True if path is a file, False otherwise.
        """
        try:
            return self.info(path)["type"] == "file"
        except OSError:
            return False

    def url(self, path: str) -> str:
        """
        Get the HTTP URL of the given path.

        Args:
            path (`str`):
                Path to get URL for.

        Returns:
            `str`: HTTP URL to access the file or directory on the Hub.
        """
        resolved_path = self.resolve_path(path)
        if isinstance(resolved_path, HfFileSystemResolvedBucketPath):
            url = f"{self.endpoint}/buckets/{resolved_path.bucket_id}/resolve/{quote(resolved_path.path)}"
        else:
            url = hf_hub_url(
                resolved_path.repo_id,
                resolved_path.path_in_repo,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
                endpoint=self.endpoint,
            )
        if self.isdir(path):
            url = url.replace("/resolve/", "/tree/", 1)
        return url

    def _get_xet_file_data(
        self, resolved_path: HfFileSystemResolvedRepositoryPath | HfFileSystemResolvedBucketPath, info: dict[str, Any]
    ) -> XetFileData | None:
        """Build the Xet metadata of a file from its `info()` entry, or return None if it can't be downloaded with Xet."""
        if not is_xet_available() or (xet_hash := info.get("xet_hash")) is None:
            return None
        if isinstance(resolved_path, HfFileSystemResolvedBucketPath):
            repo_id, repo_type, revision = resolved_path.bucket_id, "bucket", None
        else:
            # Special refs (e.g. "refs/pr/1") must be quoted in the token route
            repo_id, repo_type = resolved_path.repo_id, resolved_path.repo_type
            revision = quote(resolved_path.revision, safe="")
        refresh_route = xet_connection_info_refresh_url(
            token_type=XetTokenType.READ,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            endpoint=self.endpoint,
        )
        return XetFileData(file_hash=xet_hash, refresh_route=refresh_route)

    def get_file(self, rpath, lpath=None, callback=_DEFAULT_CALLBACK, outfile=None, **kwargs) -> None:
        """
        Copy single remote file to local.

        > [!WARNING]
        > Note: When possible, use `HfApi.hf_hub_download()` or `HfApi.download_bucket_files` for better performance.

        Args:
            rpath (`str`):
                Remote path to download from.
            lpath (`str`, *optional*):
                Local path to download to. Can be omitted if `outfile` is provided.
            callback (`Callback`, *optional*):
                Optional callback to track download progress. Defaults to no callback.
            outfile (`IO`, *optional*):
                Optional file-like object to write to. If provided, `lpath` is ignored.

        """
        revision = kwargs.get("revision")
        resolve_remote_path = self.resolve_path(rpath, revision=revision)
        # Recursive downloads map remote filenames to local paths, including on Windows.
        # Validate before creating directories, opening files, or delegating to fsspec.
        _validate_relative_filename(resolve_remote_path.path)
        unhandled_kwargs = set(kwargs.keys()) - {"revision"}
        if not isinstance(callback, (NoOpCallback, TqdmCallback)) or len(unhandled_kwargs) > 0:
            # for now, let's not handle custom callbacks
            # and let's not handle custom kwargs
            return super().get_file(rpath, lpath, callback=callback, outfile=outfile, **kwargs)

        info = self.info(rpath, revision=revision)

        # Adapted from https://github.com/fsspec/filesystem_spec/blob/0f76baaae4a227b085b18e8469e6cc1792a95ade/fsspec/spec.py#L975
        close_file = False
        if outfile is None:
            if lpath is None:
                raise ValueError("Either `lpath` or `outfile` must be provided.")
            if isfilelike(lpath):
                outfile = lpath
            elif self.isdir(rpath):
                os.makedirs(lpath, exist_ok=True)
                return None
            else:
                os.makedirs(os.path.dirname(lpath) or os.curdir, exist_ok=True)
                if (xet_file_data := self._get_xet_file_data(resolve_remote_path, info)) is not None:
                    # hf_xet writes the file to disk directly
                    callback.set_size(info["size"])
                    xet_get(
                        incomplete_path=Path(lpath),
                        xet_file_data=xet_file_data,
                        headers=self._api._build_hf_headers(),
                        expected_size=info["size"],
                        displayed_filename=rpath,
                        _tqdm_bar=callback.tqdm if isinstance(callback, TqdmCallback) else None,
                    )
                    return None
                outfile = open(lpath, "wb")
                close_file = True
        initial_pos = outfile.tell()

        # Custom implementation of `get_file` to use `http_get`.
        expected_size = info["size"]
        callback.set_size(expected_size)
        try:
            http_get(
                url=self.url(resolve_remote_path.unresolve()),
                temp_file=outfile,  # type: ignore
                displayed_filename=rpath,
                expected_size=expected_size,
                resume_size=0,
                headers=self._api._build_hf_headers(),
                _tqdm_bar=callback.tqdm if isinstance(callback, TqdmCallback) else None,
            )
            outfile.seek(initial_pos)
        finally:
            # Close file only if we opened it ourselves
            if close_file:
                outfile.close()

    @property
    def transaction(self):
        """A context within which files are committed together upon exit

        Requires the file class to implement `.commit()` and `.discard()`
        for the normal and exception cases.
        """
        # Taken from https://github.com/fsspec/filesystem_spec/blob/3fbb6fee33b46cccb015607630843dea049d3243/fsspec/spec.py#L231
        # See https://github.com/huggingface/huggingface_hub/issues/1733
        raise NotImplementedError("Transactional commits are not supported.")

    def start_transaction(self):
        """Begin write transaction for deferring files, non-context version"""
        # Taken from https://github.com/fsspec/filesystem_spec/blob/3fbb6fee33b46cccb015607630843dea049d3243/fsspec/spec.py#L241
        # See https://github.com/huggingface/huggingface_hub/issues/1733
        raise NotImplementedError("Transactional commits are not supported.")

    def __reduce__(self):
        # re-populate the instance cache at HfFileSystem._cache and re-populate the state of every instance
        return make_instance, (
            type(self),
            self.storage_args,
            self.storage_options,
            self._get_instance_state(),
        )

    def _get_instance_state(self):
        return {
            "dircache": deepcopy(self.dircache),
            "_repo_and_revision_exists_cache": deepcopy(self._repo_and_revision_exists_cache),
            "_bucket_exists_cache": deepcopy(self._bucket_exists_cache),
        }


class HfFileSystemFile(fsspec.spec.AbstractBufferedFile):
    def __init__(self, fs: HfFileSystem, path: str, revision: str | None = None, **kwargs):
        try:
            self.resolved_path = fs.resolve_path(path, revision=revision)
        except FileNotFoundError as e:
            if "w" in kwargs.get("mode", ""):
                raise FileNotFoundError(
                    f"{e}.\nMake sure the repository and revision exist before writing data."
                ) from e
            raise
        super().__init__(fs, self.resolved_path.unresolve(), **kwargs)
        self.fs: HfFileSystem

    def __del__(self):
        if not hasattr(self, "resolved_path"):
            # Means that the constructor failed. Nothing to do.
            return
        return super().__del__()

    def _fetch_range(self, start: int, end: int) -> bytes:
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        url = self.url()
        r = http_backoff("GET", url, headers=headers, timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT)
        hf_raise_for_status(r)
        return r.content

    def _initiate_upload(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(prefix="hffs-", delete=False)

    def _upload_chunk(self, final: bool = False) -> None:
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            self.temp_file.close()
            if isinstance(self.resolved_path, HfFileSystemResolvedBucketPath):
                self.fs._api.batch_bucket_files(
                    self.resolved_path.bucket_id, add=[(self.temp_file.name, self.resolved_path.path)]
                )
            else:
                self.fs._api.upload_file(
                    path_or_fileobj=self.temp_file.name,
                    path_in_repo=self.resolved_path.path_in_repo,
                    repo_id=self.resolved_path.repo_id,
                    token=self.fs.token,
                    repo_type=self.resolved_path.repo_type,
                    revision=self.resolved_path.revision,
                    commit_message=self.kwargs.get("commit_message"),
                    commit_description=self.kwargs.get("commit_description"),
                )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache(
                path=self.resolved_path.unresolve(),
            )

    def read(self, length=-1):
        """Read remote file.

        If `length` is not provided or is -1, the entire file is downloaded and read. On POSIX systems the file is
        loaded in memory directly. Otherwise, the file is downloaded to a temporary file and read from there.
        """
        if self.mode == "rb" and (length is None or length == -1) and self.loc == 0:
            with self.fs.open(self.path, "rb", block_size=0) as f:  # block_size=0 enables fast streaming
                out = f.read()
                self.loc += len(out)
                return out
        return super().read(length)

    def url(self) -> str:
        return self.fs.url(self.path)


class HfFileSystemStreamFile(fsspec.spec.AbstractBufferedFile):
    def __init__(
        self,
        fs: HfFileSystem,
        path: str,
        mode: str = "rb",
        revision: str | None = None,
        block_size: int = 0,
        cache_type: str = "none",
        **kwargs,
    ):
        if block_size != 0:
            raise ValueError(f"HfFileSystemStreamFile only supports block_size=0 but got {block_size}")
        if cache_type != "none":
            raise ValueError(f"HfFileSystemStreamFile only supports cache_type='none' but got {cache_type}")
        if "w" in mode:
            raise ValueError(f"HfFileSystemStreamFile only supports reading but got mode='{mode}'")
        # Let a missing repo/revision/file surface as FileNotFoundError (as the buffered
        # `HfFileSystemFile` does) instead of an AttributeError on `self.resolved_path` below.
        self.resolved_path = fs.resolve_path(path, revision=revision)
        # avoid an unnecessary .info() call to instantiate .details
        self.details = {"name": self.resolved_path.unresolve(), "size": None}
        super().__init__(
            fs, self.resolved_path.unresolve(), mode=mode, block_size=block_size, cache_type=cache_type, **kwargs
        )
        self.response: httpx2.Response | None = None
        self.fs: HfFileSystem
        self._exit_stack = ExitStack()
        # streaming state
        self._stream_iterator: Iterator[bytes] | None = None
        self._stream_buffer = bytearray()

    def seek(self, loc: int, whence: int = 0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError("Cannot seek streaming HF file")

    def read(self, length: int = -1):
        """Read the remote file.

        If the file is already open, we reuse the connection.
        Otherwise, open a new connection and read from it.

        If reading the stream fails, we retry with a new connection.
        """
        if self.response is None:
            self._open_connection()

        retried_once = False
        while True:
            try:
                if self.response is None or self._stream_iterator is None:
                    return b""  # Already read the entire file
                out = self._read_from_stream(self._stream_iterator, length)
                self.loc += len(out)
                return out
            except Exception:
                if self.response is not None:
                    self.response.close()
                if retried_once:  # Already retried once, give up
                    raise
                # First failure, retry with range header
                self._open_connection()
                retried_once = True

    def _read_from_stream(self, iterator: Iterator[bytes], length: int = -1) -> bytes:
        """Read up to `length` bytes from stream buffer and stream.

        If length < 0, read until EOF.

        If EOF is reached before length, fewer bytes may be returned.
        """
        if length == 0:
            return b""

        if length < 0:
            buf = bytearray(self._stream_buffer)
            self._stream_buffer.clear()
            for chunk in iterator:
                buf.extend(chunk)
            return bytes(buf)

        if length <= len(self._stream_buffer):
            result = bytes(self._stream_buffer[:length])
            del self._stream_buffer[:length]
            return result

        buf = bytearray(self._stream_buffer)
        self._stream_buffer.clear()
        for chunk in iterator:
            need = length - len(buf)
            if need > len(chunk):
                buf.extend(chunk)
            else:
                buf.extend(chunk[:need])
                self._stream_buffer.extend(chunk[need:])
                break
        return bytes(buf)

    def url(self) -> str:
        return self.fs.url(self.path)

    def __del__(self):
        if not hasattr(self, "resolved_path"):
            # Means that the constructor failed. Nothing to do.
            return
        self._exit_stack.close()
        return super().__del__()

    def __reduce__(self):
        return reopen, (self.fs, self.path, self.mode, self.blocksize, self.cache.name)

    def _open_connection(self):
        """Open a connection to the remote file."""
        # reset streaming state
        self._stream_buffer.clear()
        self._stream_iterator = None

        url = self.url()
        headers = self.fs._api._build_hf_headers()
        if self.loc > 0:
            headers["Range"] = f"bytes={self.loc}-"
        self.response = self._exit_stack.enter_context(
            http_stream_backoff(
                "GET",
                url,
                headers=headers,
                timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
            )
        )

        try:
            hf_raise_for_status(self.response)
        except HfHubHTTPError as e:
            if e.response.status_code == 416:
                # Range not satisfiable => means that we have already read the entire file
                self.response = None
                return
            raise

        self._stream_iterator = self.response.iter_bytes()


def safe_revision(revision: str) -> str:
    return revision if SPECIAL_REFS_REVISION_REGEX.search(revision) else safe_quote(revision)


def safe_quote(s: str) -> str:
    return quote(s, safe="")


def _raise_file_not_found(path: str, err: Exception | None) -> NoReturn:
    msg = path
    if isinstance(err, RepositoryNotFoundError):
        msg = f"{path} (repository not found)"
    elif isinstance(err, RevisionNotFoundError):
        msg = f"{path} (revision not found)"
    elif isinstance(err, HFValidationError):
        msg = f"{path} (invalid repository id)"
    raise FileNotFoundError(msg) from err


def reopen(fs: HfFileSystem, path: str, mode: str, block_size: int, cache_type: str):
    return fs.open(path, mode=mode, block_size=block_size, cache_type=cache_type)


def make_instance(cls, args, kwargs, instance_state):
    fs = cls(*args, **kwargs)
    for attr, state_value in instance_state.items():
        setattr(fs, attr, state_value)
    return fs


hffs = HfFileSystem()
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
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from . import constants
from .utils import get_token


if TYPE_CHECKING:
    import duckdb


@dataclass(frozen=True)
class DatasetParquetEntry:
    """Represents a single parquet file available for a dataset on the Hub."""

    config: str
    split: str
    url: str
    size: int


def execute_raw_sql_query(sql_query: str, *, token: str | bool | None = None) -> list[dict[str, Any]]:
    normalized_query = sql_query.strip().rstrip(";").strip()
    _raise_on_forbidden_query(normalized_query)

    connection = None
    try:
        connection = _get_duckdb_connection(token=token)
        relation = connection.sql(normalized_query)
        if relation is None:
            raise ValueError("SQL query must return rows.")

        if isinstance(relation, _DuckDBCliRelation):
            # DuckDB binary => run CLI => parse JSON
            return relation.execute()
        else:
            # DuckDB Python API => fetch columns + rows => convert to dicts
            columns = tuple(column[0] for column in relation.description)
            rows = tuple(tuple(row) for row in relation.fetchall())
            return [dict(zip(columns, row)) for row in rows]
    finally:
        if connection is not None:
            connection.close()


def _raise_on_forbidden_query(query: str) -> None:
    if len(query) == 0:
        raise ValueError("SQL query cannot be empty.")

    # DuckDB CLI meta-commands are dot-prefixed words (e.g. `.shell`, `.output`).
    # Let's forbid them for now but allow SQL expressions like `.5` that can legitimately start a line.
    for line in query.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(".") and stripped[1:2].isalpha():
            raise ValueError("DuckDB CLI meta-commands are not allowed in SQL queries.")


def _get_duckdb_connection(
    token: str | bool | None,
) -> Union["duckdb.DuckDBPyConnection", "_DuckDBCliConnection"]:
    try:
        # If DuckDB is installed as a Python package, use it!
        import duckdb
    except ImportError as error:
        # Otherwise, use the DuckDB CLI binary.
        duckdb_binary = shutil.which("duckdb")
        if duckdb_binary is None:
            raise ImportError(
                "DuckDB is required for `hf datasets sql`. Install the Python package with `pip install duckdb` or "
                "install the DuckDB CLI binary (for example `brew install duckdb`)."
            ) from error
        return _DuckDBCliConnection(binary_path=duckdb_binary, token=token)

    # Create a new connection (Python API).
    connection = duckdb.connect()
    try:
        for statement in _build_duckdb_secret_statements(token):
            connection.execute(statement)
        return connection
    except Exception:
        connection.close()
        raise


@dataclass
class _DuckDBCliConnection:
    """DuckDB connection.

    Mimics the DuckDB Python API, but runs the queries via the DuckDB CLI binary.
    """

    binary_path: str
    token: str | bool | None

    def __post_init__(self) -> None:
        self._setup_statements = _build_duckdb_secret_statements(self.token)

    def sql(self, query: str) -> "_DuckDBCliRelation":
        return _DuckDBCliRelation(binary_path=self.binary_path, setup_statements=self._setup_statements, query=query)

    def close(self) -> None:
        pass


@dataclass
class _DuckDBCliRelation:
    """DuckDB relation.

    Mimics the DuckDB Python API, but runs the queries via the DuckDB CLI binary.
    """

    binary_path: str
    setup_statements: list[str]
    query: str

    def execute(self) -> list[dict[str, Any]]:
        # Build the DuckDB CLI input.
        setup = []
        if self.setup_statements:
            setup = [
                f".output {os.devnull}",
                *(f"{stmt};" for stmt in self.setup_statements),
                ".output",
            ]
        full_query = "\n".join(setup + [self.query + ";"])

        # Run DuckDB binary
        result = subprocess.run(
            [self.binary_path, "-json"],
            input=full_query,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            error_message = result.stderr.strip() or result.stdout.strip() or "DuckDB CLI command failed."
            raise RuntimeError(error_message)

        # Parse JSON output and return
        return json.loads(result.stdout.strip())


def _build_duckdb_secret_statements(token: str | bool | None) -> list[str]:
    if token is None or token is True:
        token = get_token()

    if not token:
        return []

    escaped_token = token.replace("'", "''")
    escaped_endpoint = constants.ENDPOINT.replace("'", "''")
    return [
        f"CREATE OR REPLACE SECRET hf_hub_token (TYPE HTTP, BEARER_TOKEN '{escaped_token}', SCOPE '{escaped_endpoint}')",
        f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{escaped_token}')",
    ]
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
"""Git LFS related type definitions and utilities"""

import io
import re
from collections.abc import Iterable
from math import ceil
from os.path import getsize
from typing import TYPE_CHECKING, BinaryIO, TypedDict
from urllib.parse import unquote

from huggingface_hub import constants

from .utils import (
    build_hf_headers,
    fix_hf_endpoint_in_url,
    hf_raise_for_status,
    http_backoff,
    logging,
    validate_hf_hub_args,
)
from .utils._lfs import SliceFileObj
from .utils.sha import sha256, sha_fileobj


if TYPE_CHECKING:
    from ._commit_api import CommitOperationAdd

logger = logging.get_logger(__name__)

OID_REGEX = re.compile(r"^[0-9a-f]{40}$")

LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"

LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


class UploadInfo:
    """
    Data structure holding required information to determine whether a blob
    should be uploaded to the hub using the LFS protocol or the regular protocol.

    The SHA256 of the blob is computed lazily: creating an `UploadInfo` from a local path only reads
    the first 512 bytes of the file. The full file is read (and hashed) only if `sha256` is accessed
    before it has been set. When a file is uploaded through the Xet protocol, the SHA256 is computed
    during upload (single read pass) and set afterwards.

    Args:
        size (`int`):
            Size in bytes of the blob
        sample (`bytes`):
            First 512 bytes of the blob
        sha256 (`bytes`, *optional*):
            SHA256 hash of the blob, if already known. Otherwise computed lazily from `source_path`.
        source_path (`str`, *optional*):
            Path to the local file the blob comes from. Required to lazily compute `sha256` if not provided.
    """

    def __init__(
        self,
        size: int,
        sample: bytes,
        sha256: bytes | None = None,
        source_path: str | None = None,
    ):
        if sha256 is None and source_path is None:
            raise ValueError("Either `sha256` or `source_path` must be provided.")
        self.size = size
        self.sample = sample
        self._sha256 = sha256
        self._source_path = source_path

    @property
    def sha256(self) -> bytes:
        """SHA256 of the blob. If not set yet, reads the whole file from `source_path` to compute it."""
        if self._sha256 is None:
            assert self._source_path is not None  # guaranteed by __init__
            with open(self._source_path, "rb") as file:
                self._sha256 = sha_fileobj(file)
        return self._sha256

    @sha256.setter
    def sha256(self, value: bytes) -> None:
        self._sha256 = value

    @property
    def is_hashed(self) -> bool:
        """Whether the SHA256 is already known (accessing `sha256` will not trigger a file read)."""
        return self._sha256 is not None

    def __repr__(self) -> str:
        sha = self._sha256.hex() if self._sha256 is not None else "<not computed>"
        return f"UploadInfo(size={self.size}, sha256={sha})"

    @classmethod
    def from_path(cls, path: str):
        size = getsize(path)
        with open(path, "rb") as file:
            sample = file.peek(512)[:512]
        return cls(size=size, sample=sample, source_path=path)

    @classmethod
    def from_bytes(cls, data: bytes):
        sha = sha256(data).digest()
        return cls(size=len(data), sample=data[:512], sha256=sha)

    @classmethod
    def from_fileobj(cls, fileobj: BinaryIO):
        sample = fileobj.read(512)
        fileobj.seek(0, io.SEEK_SET)
        sha = sha_fileobj(fileobj)
        size = fileobj.tell()
        fileobj.seek(0, io.SEEK_SET)
        return cls(size=size, sha256=sha, sample=sample)


@validate_hf_hub_args
def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: str | None,
    repo_type: str,
    repo_id: str,
    revision: str | None = None,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    transfers: list[str] | None = None,
) -> tuple[list[dict], list[dict], str | None]:
    """
    Requests the LFS batch endpoint to retrieve upload instructions

    Learn more: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        upload_infos (`Iterable` of `UploadInfo`):
            `UploadInfo` for the files that are being uploaded, typically obtained
            from `CommitOperationAdd.upload_info`
        token (`str` or `None`):
            An authentication token (see https://huggingface.co/settings/token).
            Pass `None` to fall back to the local cached token (or no token if unauthenticated).
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        revision (`str`, *optional*):
            The git revision to upload to.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.
        headers (`dict`, *optional*):
            Additional headers to include in the request
        transfers (`list`, *optional*):
            List of transfer methods to use. Defaults to ["basic", "multipart"].

    Returns:
        `LfsBatchInfo`: 3-tuple:
            - First element is the list of upload instructions from the server
            - Second element is a list of errors, if any
            - Third element is the chosen transfer adapter if provided by the server (e.g. "basic", "multipart", "xet")

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If an argument is invalid or the server response is malformed.
        [`HfHubHTTPError`]
            If the server returned an error.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    url_prefix = ""
    if repo_type in constants.REPO_TYPES_URL_PREFIXES:
        url_prefix = constants.REPO_TYPES_URL_PREFIXES[repo_type]
    batch_url = f"{endpoint}/{url_prefix}{repo_id}.git/info/lfs/objects/batch"
    payload: dict = {
        "operation": "upload",
        "transfers": transfers if transfers is not None else ["basic", "multipart"],
        "objects": [
            {
                "oid": upload.sha256.hex(),
                "size": upload.size,
            }
            for upload in upload_infos
        ],
        "hash_algo": "sha256",
    }
    if revision is not None:
        payload["ref"] = {"name": unquote(revision)}  # revision has been previously 'quoted'

    headers = {
        **LFS_HEADERS,
        **build_hf_headers(token=token),
        **(headers or {}),
    }
    resp = http_backoff("POST", batch_url, headers=headers, json=payload)
    hf_raise_for_status(resp)
    batch_info = resp.json()

    objects = batch_info.get("objects", None)
    if not isinstance(objects, list):
        raise ValueError("Malformed response from server")

    chosen_transfer = batch_info.get("transfer")
    chosen_transfer = chosen_transfer if isinstance(chosen_transfer, str) else None

    return (
        [_validate_batch_actions(obj) for obj in objects if "error" not in obj],
        [_validate_batch_error(obj) for obj in objects if "error" in obj],
        chosen_transfer,
    )


class PayloadPartT(TypedDict):
    partNumber: int
    etag: str


class CompletionPayloadT(TypedDict):
    """Payload that will be sent to the Hub when uploading multi-part."""

    oid: str
    parts: list[PayloadPartT]


def lfs_upload(
    operation: "CommitOperationAdd",
    lfs_batch_action: dict,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
) -> None:
    """
    Handles uploading a given object to the Hub with the LFS protocol.

    Can be a No-op if the content of the file is already present on the hub large file storage.

    Args:
        operation (`CommitOperationAdd`):
            The add operation triggering this upload.
        lfs_batch_action (`dict`):
            Upload instructions from the LFS batch endpoint for this object. See [`~utils.lfs.post_lfs_batch_info`] for
            more details.
        token (`str`, *optional*):
            An authentication token (see https://huggingface.co/settings/token). Used to call the
            optional LFS verify step at the end of the upload. If `None`, falls back to the local
            cached token.
        headers (`dict`, *optional*):
            Headers to include in the request, including authentication and user agent headers.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `lfs_batch_action` is improperly formatted
        [`HfHubHTTPError`]
            If the upload resulted in an error
    """
    # 0. If LFS file is already present, skip upload
    _validate_batch_actions(lfs_batch_action)
    actions = lfs_batch_action.get("actions")
    if actions is None:
        # The file was already uploaded
        logger.debug(f"Content of file {operation.path_in_repo} is already present upstream - skipping upload")
        return

    # 1. Validate server response (check required keys in dict)
    upload_action = lfs_batch_action["actions"]["upload"]
    _validate_lfs_action(upload_action)
    verify_action = lfs_batch_action["actions"].get("verify")
    if verify_action is not None:
        _validate_lfs_action(verify_action)

    # 2. Upload file (either single part or multi-part)
    header = upload_action.get("header", {})
    chunk_size = header.get("chunk_size")
    upload_url = fix_hf_endpoint_in_url(upload_action["href"], endpoint=endpoint)
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
        except (ValueError, TypeError):
            raise ValueError(
                f"Malformed response from LFS batch endpoint: `chunk_size` should be an integer. Got '{chunk_size}'."
            )
        _upload_multi_part(operation=operation, header=header, chunk_size=chunk_size, upload_url=upload_url)
    else:
        _upload_single_part(operation=operation, upload_url=upload_url)

    # 3. Verify upload went well
    if verify_action is not None:
        _validate_lfs_action(verify_action)
        verify_url = fix_hf_endpoint_in_url(verify_action["href"], endpoint)
        verify_resp = http_backoff(
            "POST",
            verify_url,
            headers=build_hf_headers(token=token, headers=headers),
            json={"oid": operation.upload_info.sha256.hex(), "size": operation.upload_info.size},
        )
        hf_raise_for_status(verify_resp)
    logger.debug(f"{operation.path_in_repo}: Upload successful")


def _validate_lfs_action(lfs_action: dict):
    """validates response from the LFS batch endpoint"""
    if not (
        isinstance(lfs_action.get("href"), str)
        and (lfs_action.get("header") is None or isinstance(lfs_action.get("header"), dict))
    ):
        raise ValueError("lfs_action is improperly formatted")
    return lfs_action


def _validate_batch_actions(lfs_batch_actions: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_actions.get("oid"), str) and isinstance(lfs_batch_actions.get("size"), int)):
        raise ValueError("lfs_batch_actions is improperly formatted")

    upload_action = lfs_batch_actions.get("actions", {}).get("upload")
    verify_action = lfs_batch_actions.get("actions", {}).get("verify")
    if upload_action is not None:
        _validate_lfs_action(upload_action)
    if verify_action is not None:
        _validate_lfs_action(verify_action)
    return lfs_batch_actions


def _validate_batch_error(lfs_batch_error: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_error.get("oid"), str) and isinstance(lfs_batch_error.get("size"), int)):
        raise ValueError("lfs_batch_error is improperly formatted")
    error_info = lfs_batch_error.get("error")
    if not (
        isinstance(error_info, dict)
        and isinstance(error_info.get("message"), str)
        and isinstance(error_info.get("code"), int)
    ):
        raise ValueError("lfs_batch_error is improperly formatted")
    return lfs_batch_error


def _upload_single_part(operation: "CommitOperationAdd", upload_url: str) -> None:
    """
    Uploads `fileobj` as a single PUT HTTP request (basic LFS transfer protocol)

    Args:
        upload_url (`str`):
            The URL to PUT the file to.
        fileobj:
            The file-like object holding the data to upload.

    Raises:
        [`HfHubHTTPError`]
            If the upload resulted in an error.
    """
    with operation.as_file(with_tqdm=True) as fileobj:
        # S3 might raise a transient 500 error -> let's retry if that happens
        response = http_backoff("PUT", upload_url, data=fileobj)
        hf_raise_for_status(response)


def _upload_multi_part(operation: "CommitOperationAdd", header: dict, chunk_size: int, upload_url: str) -> None:
    """
    Uploads file using HF multipart LFS transfer protocol.
    """
    # 1. Get upload URLs for each part
    sorted_parts_urls = _get_sorted_parts_urls(header=header, upload_info=operation.upload_info, chunk_size=chunk_size)

    # 2. Upload parts (pure Python)
    response_headers = _upload_parts_iteratively(
        operation=operation, sorted_parts_urls=sorted_parts_urls, chunk_size=chunk_size
    )

    # 3. Send completion request
    # NOTE: `upload_url` is the Hub completion endpoint (not the S3 upload URLs).
    completion_res = http_backoff(
        "POST",
        upload_url,
        json=_get_completion_payload(response_headers, operation.upload_info.sha256.hex()),
        headers=LFS_HEADERS,
    )
    hf_raise_for_status(completion_res)


def _get_sorted_parts_urls(header: dict, upload_info: UploadInfo, chunk_size: int) -> list[str]:
    sorted_part_upload_urls = [
        upload_url
        for _, upload_url in sorted(
            [
                (int(part_num, 10), upload_url)
                for part_num, upload_url in header.items()
                if part_num.isdigit() and len(part_num) > 0
            ],
            key=lambda t: t[0],
        )
    ]
    num_parts = len(sorted_part_upload_urls)
    if num_parts != ceil(upload_info.size / chunk_size):
        raise ValueError("Invalid server response to upload large LFS file")
    return sorted_part_upload_urls


def _get_completion_payload(response_headers: list[dict], oid: str) -> CompletionPayloadT:
    parts: list[PayloadPartT] = []
    for part_number, header in enumerate(response_headers):
        etag = header.get("etag")
        if etag is None or etag == "":
            raise ValueError(f"Invalid etag (`{etag}`) returned for part {part_number + 1}")
        parts.append(
            {
                "partNumber": part_number + 1,
                "etag": etag,
            }
        )
    return {"oid": oid, "parts": parts}


def _upload_parts_iteratively(
    operation: "CommitOperationAdd", sorted_parts_urls: list[str], chunk_size: int
) -> list[dict]:
    headers = []
    with operation.as_file(with_tqdm=True) as fileobj:
        for part_idx, part_upload_url in enumerate(sorted_parts_urls):
            with SliceFileObj(
                fileobj,
                seek_from=chunk_size * part_idx,
                read_limit=chunk_size,
            ) as fileobj_slice:
                # S3 might raise a transient 500 error -> let's retry if that happens
                part_upload_res = http_backoff("PUT", part_upload_url, data=fileobj_slice)
                hf_raise_for_status(part_upload_res)
                headers.append(part_upload_res.headers)
    return headers  # type: ignore
<|endoftext|>import copy
import errno
import os
import re
import shutil
import stat
import time
import uuid
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, NoReturn, cast, overload
from urllib.parse import quote

import httpx2
from tqdm import tqdm as base_tqdm

from . import constants
from ._local_folder import (
    _create_cachedir_tag,
    _validate_relative_filename,
    get_local_download_paths,
    read_download_metadata,
    write_download_metadata,
)
from ._revision import ResolvedRevision
from ._tree_cache import read_tree_cache, tree_cache_folder_for_local_dir
from .errors import (
    FileMetadataError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from .utils import (
    OfflineModeIsEnabled,
    SoftTemporaryDirectory,
    WeakFileLock,
    XetFileData,
    build_hf_headers,
    hf_raise_for_status,
    logging,
    parse_xet_file_data_from_response,
    validate_hf_hub_args,
)
from .utils._http import (
    _DEFAULT_RETRY_ON_EXCEPTIONS,
    _DEFAULT_RETRY_ON_STATUS_CODES,
    _adjust_range_header,
    _httpx2_follow_hub_redirects_with_backoff,
    _is_same_or_hub_host,
    flag_as_download_call,
    http_stream_backoff,
)
from .utils._paths import as_extended_path
from .utils._runtime import is_xet_available
from .utils._shared_blobs import (
    has_shared_blob,
    publish_blob_to_shared_store,
    shared_blobs_enabled,
    try_link_from_shared_store,
)
from .utils._xet import XetTokenType, is_valid_xet_hash, xet_connection_info_refresh_url
from .utils.sha import sha_fileobj
from .utils.tqdm import _get_progress_bar_context


logger = logging.get_logger(__name__)

# Return value when trying to load a file from cache but the file does not exist in the distant repo.
_CACHED_NO_EXIST = object()
_CACHED_NO_EXIST_T = Any

# Regex to get filename from a "Content-Disposition" header for CDN-served files
HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)";')

# Regex to check if the revision IS directly a commit_hash
REGEX_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")

# Regex to check if the file etag IS a valid sha256
REGEX_SHA256 = re.compile(r"[0-9a-f]{64}")

_are_symlinks_supported_in_dir: dict[str, bool] = {}

# Internal retry timeout for metadata fetch when no local file exists
_ETAG_RETRY_TIMEOUT = 60


def are_symlinks_supported(cache_dir: str | Path | None = None) -> bool:
    """Return whether the symlinks are supported on the machine.

    Since symlinks support can change depending on the mounted disk, we need to check
    on the precise cache folder. By default, the default HF cache directory is checked.

    Args:
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.

    Returns: [bool] Whether symlinks are supported in the directory.
    """
    # Defaults to HF cache
    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())  # make it unique

    # If symlinks are explicitly disabled by the user, always return False
    if constants.HF_HUB_DISABLE_SYMLINKS:
        return False

    # Cache symlink compatibility per directory, publishing the result only after the probe completes.
    if cache_dir not in _are_symlinks_supported_in_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with SoftTemporaryDirectory(dir=cache_dir) as tmpdir:
            src_path = Path(tmpdir) / "dummy_file_src"
            src_path.touch()
            dst_path = Path(tmpdir) / "dummy_file_dst"

            # Relative source path as in `_create_symlink``
            relative_src = os.path.relpath(src_path, start=os.path.dirname(dst_path))
            try:
                os.symlink(relative_src, dst_path)
                _are_symlinks_supported_in_dir[cache_dir] = True
            except OSError:
                # Likely running on Windows
                _are_symlinks_supported_in_dir[cache_dir] = False

                if not constants.HF_HUB_DISABLE_SYMLINKS_WARNING:
                    message = (
                        "`huggingface_hub` cache-system uses symlinks by default to"
                        " efficiently store duplicated files but your machine does not"
                        f" support them in {cache_dir}. Caching files will still work"
                        " but in a degraded version that might require more space on"
                        " your disk. This warning can be disabled by setting the"
                        " `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For"
                        " more details, see"
                        " https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations."
                    )
                    if os.name == "nt":
                        message += (
                            "\nTo support symlinks on Windows, you either need to"
                            " activate Developer Mode or to run Python as an"
                            " administrator. In order to activate developer mode,"
                            " see this article:"
                            " https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development"
                        )
                    warnings.warn(message)

    return _are_symlinks_supported_in_dir[cache_dir]


@dataclass(frozen=True)
class HfFileMetadata:
    """Data structure containing information about a file versioned on the Hub.

    Returned by [`get_hf_file_metadata`] based on a URL.

    Args:
        commit_hash (`str`, *optional*):
            The commit_hash related to the file.
        etag (`str`, *optional*):
            Etag of the file on the server.
        location (`str`):
            Location where to download the file. Can be a Hub url or not (CDN).
        size (`size`):
            Size of the file. In case of an LFS file, contains the size of the actual
            LFS file, not the pointer.
        xet_file_data (`XetFileData`, *optional*):
            Xet information for the file. This is only set if the file is stored using Xet storage.
    """

    commit_hash: str | None
    etag: str | None
    location: str
    size: int | None
    xet_file_data: XetFileData | None


@dataclass
class DryRunFileInfo:
    """Information returned when performing a dry run of a file download.

    Returned by [`hf_hub_download`] when `dry_run=True`.

    Args:
        commit_hash (`str`):
            The commit_hash related to the file.
        file_size (`int`, *optional*):
            Size of the file, if known. In case of an LFS file, contains the size of the actual LFS file, not the
            pointer.
        filename (`str`):
            Name of the file in the repo.
        is_cached (`bool`):
            Whether the file is already cached locally.
        will_download (`bool`):
            Whether the file will be downloaded if `hf_hub_download` is called with `dry_run=False`.
            In practice, will_download is `True` if the file is not cached or if `force_download=True`.
    """

    commit_hash: str
    file_size: int | None
    filename: str
    local_path: str
    is_cached: bool
    will_download: bool


@validate_hf_hub_args
def hf_hub_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Construct the URL of a file from the given information.

    The resolved address can either be a huggingface.co-hosted url, or a link to
    Cloudfront (a Content Delivery Network, or CDN) for large files which are
    more than a few MBs.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) name and a repo name separated
            by a `/`.
        filename (`str`):
            The name of the file in the repo.
        subfolder (`str`, *optional*):
            An optional value corresponding to a folder inside the repo.
        repo_type (`str`, *optional*):
            Set to `"dataset"`, `"space"` or `"kernel"` if downloading from a dataset, space or kernel repo,
            `None` or `"model"` if downloading from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.

    Example:

    ```python
    >>> from huggingface_hub import hf_hub_url

    >>> hf_hub_url(
    ...     repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin"
    ... )
    'https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin'
    ```

    > [!TIP]
    > Notes:
    >
    >     Cloudfront is replicated over the globe so downloads are way faster for
    >     the end user (and it also lowers our bandwidth costs).
    >
    >     Cloudfront aggressively caches files by default (default TTL is 24
    >     hours), however this is not an issue here because we implement a
    >     git-based versioning system on huggingface.co, which means that we store
    >     the files on S3/Cloudfront in a content-addressable way (i.e., the file
    >     name is its hash). Using content-addressable filenames means cache can't
    >     ever be stale.
    >
    >     In terms of client-side caching from this library, we base our caching
    >     on the objects' entity tag (`ETag`), which is an identifier of a
    >     specific version of a resource [1]_. An object's ETag is: its git-sha1
    >     if stored in git, or its sha256 if stored in git-lfs.

    References:

    -  [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
    """
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError("Invalid repo type")

    if repo_type in constants.REPO_TYPES_URL_PREFIXES:
        repo_id = constants.REPO_TYPES_URL_PREFIXES[repo_type] + repo_id  # type: ignore

    if revision is None:
        revision = constants.DEFAULT_REVISION
    url = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
        repo_id=repo_id, revision=quote(revision, safe=""), filename=quote(filename)
    )
    # Update endpoint if provided
    if endpoint is not None and url.startswith(constants.ENDPOINT):
        url = endpoint + url[len(constants.ENDPOINT) :]
    return url


def _get_file_length_from_http_response(response: httpx2.Response) -> int | None:
    """
    Get the length of the file from the HTTP response headers.

    This function extracts the file size from the HTTP response headers, either from the
    `Content-Range` or `Content-Length` header, if available (in that order).

    Args:
        response (`httpx2.Response`):
            The HTTP response object.

    Returns:
        `int` or `None`: The length of the file in bytes, or None if not available.
    """

    # If HTTP response contains compressed body (e.g. gzip), the `Content-Length` header will
    # contain the length of the compressed body, not the uncompressed file size.
    # And at the start of transmission there's no way to know the uncompressed file size for gzip,
    # thus we return None in that case.
    content_encoding = response.headers.get("Content-Encoding", "identity").lower()
    if content_encoding != "identity":
        # gzip/br/deflate/zstd etc
        return None

    content_range = response.headers.get("Content-Range")
    if content_range is not None:
        return int(content_range.rsplit("/")[-1])

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        return int(content_length)

    return None


@validate_hf_hub_args
def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    resume_size: int = 0,
    headers: dict[str, Any] | None = None,
    expected_size: int | None = None,
    displayed_filename: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    _nb_retries: int = 5,
    _tqdm_bar: base_tqdm | None = None,
) -> None:
    """
    Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.

    If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely a
    transient error (network outage?). We log a warning message and try to resume the download a few times before
    giving up. The method gives up after 5 attempts if no new data has being received from the server.

    Args:
        url (`str`):
            The URL of the file to download.
        temp_file (`BinaryIO`):
            The file-like object where to save the file. Content is written from the object's current position, so the
            caller may hand over a file that already contains data of its own (see [`HfFileSystem.get_file`]).
        resume_size (`int`, *optional*):
            The number of bytes already downloaded. If set to 0 (default), the whole file is download. If set to a
            positive number, the download will resume at the given position.
        headers (`dict`, *optional*):
            Dictionary of HTTP Headers to send with the request.
        expected_size (`int`, *optional*):
            The expected size of the file to download. If set, the download will raise an error if the size of the
            received content is different from the expected one.
        displayed_filename (`str`, *optional*):
            The filename of the file that is being downloaded. Value is used only to display a nice progress bar. If
            not set, the filename is guessed from the URL or the `Content-Disposition` header.
    """
    if expected_size is not None and resume_size == expected_size:
        # If the file is already fully downloaded, we don't need to download it again.
        return

    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = _adjust_range_header(headers.get("Range"), resume_size)
    elif expected_size and expected_size > constants.MAX_HTTP_DOWNLOAD_SIZE:
        # Any files over 50GB will not be available through basic http requests.
        raise ValueError(
            "The file is too large to be downloaded using the regular download method. "
            " Install `hf_xet` with `pip install hf_xet` for xet-powered downloads."
        )

    # Keep the response and progress bar open while recursive retries reuse them.
    with ExitStack() as stack:
        progress = _tqdm_bar
        new_resume_size = resume_size
        try:
            response = stack.enter_context(
                http_stream_backoff(
                    method="GET",
                    url=url,
                    headers=headers,
                    timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
                    retry_on_exceptions=(),
                    retry_on_status_codes=(408, 429),
                )
            )
            hf_raise_for_status(response)

            # If we requested a Range but got 200 back, the server ignored our Range header
            # (e.g. CloudFront with Accept-Encoding: gzip). Reset file to avoid corruption.
            if resume_size > 0 and response.status_code == 200:
                # Rewind to where this download started, which is not necessarily the start of the file: the caller
                # may have handed over a file object that already contains data of its own.
                temp_file.seek(max(temp_file.tell() - resume_size, 0))
                temp_file.truncate()
                if _tqdm_bar is not None:
                    # When the progress bar is reused across retries, its counter has already been advanced by `resume_size`
                    # worth of chunks from earlier attempts. Those bytes are gone from disk now, so roll the counter back
                    # to keep the upcoming full re-download from double-counting (e.g. ending at 130/100 on a 100-byte file).
                    _tqdm_bar.update(-resume_size)
                    if callable(update_transfer := getattr(_tqdm_bar, "update_transfer", None)):
                        update_transfer(-resume_size)
                resume_size = 0

            total: int | None = _get_file_length_from_http_response(response)
            if expected_size is None:
                expected_size = total
            elif total is None:
                # Hub serves compressible text files (e.g. vocab.json) with `Content-Encoding: gzip` and
                # `Transfer-Encoding: chunked`, so the response carries no `Content-Length`. Fall back to the caller's
                # `expected_size` (always known from the metadata HEAD on the hf_hub path) so the progress bar, and any
                # aggregating wrapper such as snapshot_download's `_AggregatedTqdm` — still sees the file size.
                total = expected_size

            if displayed_filename is None:
                displayed_filename = url
                content_disposition = response.headers.get("Content-Disposition")
                if content_disposition is not None:
                    match = HEADER_FILENAME_PATTERN.search(content_disposition)
                    if match is not None:
                        # Means file is on CDN
                        displayed_filename = match.groupdict()["filename"]

            # Truncate filename if too long to display
            if len(displayed_filename) > 40:
                displayed_filename = f"(…){displayed_filename[-40:]}"

            consistency_error_message = (
                f"Consistency check failed: file should be of size {expected_size} but has size"
                f" {{actual_size}} ({displayed_filename}).\nThis is usually due to network issues while downloading the file."
                " Please retry with `force_download=True`."
            )
            progress_cm = _get_progress_bar_context(
                desc=displayed_filename,
                log_level=logger.getEffectiveLevel(),
                total=total,
                initial=resume_size,
                name="huggingface_hub.http_get",
                tqdm_class=cast(Any, tqdm_class),
                _tqdm_bar=cast(Any, _tqdm_bar),
            )

            progress = stack.enter_context(progress_cm)
            new_resume_size = resume_size
            for chunk in response.iter_bytes(chunk_size=constants.DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    if callable(update_transfer := getattr(progress, "update_transfer", None)):
                        update_transfer(len(chunk))
                    temp_file.write(chunk)
                    new_resume_size += len(chunk)
                    # Some data has been downloaded from the server so we reset the number of retries.
                    _nb_retries = 5
        except (httpx2.ConnectError, httpx2.TimeoutException, httpx2.RemoteProtocolError) as e:
            # Retry transient failures both when opening the stream and while reading its body.
            if _nb_retries <= 0:
                logger.warning("Error while downloading from %s: %s\nMax retries exceeded.", url, str(e))
                raise
            logger.warning("Error while downloading from %s: %s\nTrying to resume download...", url, str(e))
            time.sleep(1)
            return http_get(
                url=url,
                temp_file=temp_file,
                resume_size=new_resume_size,
                headers=initial_headers,
                expected_size=expected_size,
                tqdm_class=tqdm_class,
                _nb_retries=_nb_retries - 1,
                # Reuse the existing progress bar across retries so a custom `tqdm_class` (e.g. snapshot_download's `_AggregatedTqdm`,
                # which mutates a shared parent bar in `__init__`) is not re-instantiated and does not double-count `total`/`initial`.
                _tqdm_bar=progress,
            )

    # Compare against the bytes downloaded by this call rather than the absolute file position: the two differ
    # whenever the caller passed a file object that was not positioned at 0.
    if expected_size is not None and expected_size != new_resume_size:
        raise OSError(
            consistency_error_message.format(
                actual_size=new_resume_size,
            )
        )


def xet_get(
    *,
    incomplete_path: Path,
    xet_file_data: XetFileData,
    headers: dict[str, str],
    expected_size: int | None = None,
    displayed_filename: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    _tqdm_bar: base_tqdm | None = None,
) -> None:
    """
    Download a file using Xet storage service.

    Args:
        incomplete_path (`Path`):
            The path to the file to download.
        xet_file_data (`XetFileData`):
            The file metadata needed to make the request to the xet storage service.
        headers (`dict[str, str]`):
            The headers to send to the xet storage service.
        expected_size (`int`, *optional*):
            The expected size of the file to download. If set, the download will raise an error if the size of the
            received content is different from the expected one.
        displayed_filename (`str`, *optional*):
            The filename of the file that is being downloaded. Value is used only to display a nice progress bar. If
            not set, the filename is guessed from the URL or the `Content-Disposition` header.

    **How it works:**
        The file download system uses Xet storage, which is a content-addressable storage system that breaks files into chunks
        for efficient storage and transfer.

        ``session.new_file_download_group()`` manages downloading files by:
        - Registering download tasks (each with its unique content hash) and starting download immediately in the background
        - Connecting to a storage server (CAS server) that knows how files are chunked
        - Using authentication to ensure secure access
        - Providing progress updates during download

        Authentication works transparently: the download group is seeded with a short-lived xet
        access token (fetched once per repo revision and cached across files) and accepts a
        ``token_refresh_url`` that is used to refresh it as needed.

        The download process works like this:
        1. Download tasks run in parallel:
            1.1. Prepare to write the file to disk or to a stream (e.g. truncate file, set up cache)
            1.2. Ask the server "how is this file split into chunks?" using the file's unique hash
                The server responds with:
                - Which chunks make up the complete file
                - Where each chunk can be downloaded from
            1.3. For each needed chunk:
                - Checks if we already have it in our local cache
                - If not, download it from cloud storage (S3)
                - Save it to cache for future use
                - Assemble the chunks in order to recreate the original file

    """
    try:
        from hf_xet import XetFileInfo  # type: ignore[no-redef]
    except ImportError:
        raise ValueError(
            "To use optimized download using Xet storage, you need to install the hf_xet package. "
            'Try `pip install "huggingface_hub[hf_xet]"` or `pip install hf_xet`.'
        )

    if not displayed_filename:
        displayed_filename = incomplete_path.name

    # Truncate filename if too long to display
    if len(displayed_filename) > 40:
        displayed_filename = f"{displayed_filename[:40]}(…)"

    from .utils._xet import abort_xet_session, get_xet_session, refresh_xet_connection_info, xet_headers_without_auth
    from .utils._xet_progress_reporting import XetDownloadProgressReporter

    xet_headers = xet_headers_without_auth(headers)

    # Fetched once per repo revision and cached; otherwise each download group would request its own
    # token, i.e. one Hub API call per file (rate-limited on large snapshot downloads, see #4722).
    connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

    session = get_xet_session()

    with XetDownloadProgressReporter(
        reconstruction_desc=f"{displayed_filename}: reconstructing file",
        transfer_desc=f"{displayed_filename}: downloading bytes",
        total=expected_size,
        log_level=logger.getEffectiveLevel(),
        name="huggingface_hub.xet_get",
        tqdm_class=tqdm_class,
        external_reconstruction_bar=_tqdm_bar,
    ) as progress:
        try:
            with session.new_file_download_group(
                endpoint=connection_info.endpoint,
                token=connection_info.access_token,
                token_expiry_unix_secs=connection_info.expiration_unix_epoch,
                token_refresh_url=xet_file_data.refresh_route,
                token_refresh_headers=headers,
                custom_headers=xet_headers,
                progress_callback=progress.update_progress,
            ) as group:
                group.start_download_file(
                    XetFileInfo(xet_file_data.file_hash, expected_size), str(incomplete_path.absolute())
                )
        except KeyboardInterrupt:
            abort_xet_session()
            raise


def _normalize_etag(etag: str | None) -> str | None:
    """Normalize ETag HTTP header, so it can be used to create nice filepaths.

    The HTTP spec allows two forms of ETag:
      ETag: W/"<etag_value>"
      ETag: "<etag_value>"

    For now, we only expect the second form from the server, but we want to be future-proof so we support both. For
    more context, see `TestNormalizeEtag` tests and https://github.com/huggingface/huggingface_hub/pull/1428.

    Args:
        etag (`str`, *optional*): HTTP header

    Returns:
        `str` or `None`: string that can be used as a nice directory name.
        Returns `None` if input is None.
    """
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


def _create_relative_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """Alias method used in `transformers` conversion script."""
    return _create_symlink(src=src, dst=dst, new_blob=new_blob)


def _create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """Create a symbolic link named dst pointing to src.

    By default, it will try to create a symlink using a relative path. Relative paths have 2 advantages:
    - If the cache_folder is moved (example: back-up on a shared drive), relative paths within the cache folder will
      not break.
    - Relative paths seems to be better handled on Windows. Issue was reported 3 times in less than a week when
      changing from relative to absolute paths. See https://github.com/huggingface/huggingface_hub/issues/1398,
      https://github.com/huggingface/diffusers/issues/2729 and https://github.com/huggingface/transformers/pull/22228.
      NOTE: The issue with absolute paths doesn't happen on admin mode.
    When creating a symlink from the cache to a local folder, it is possible that a relative path cannot be created.
    This happens when paths are not on the same volume. In that case, we use absolute paths.


    The result layout looks something like
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd

    If symlinks cannot be created on this platform (most likely to be Windows), the workaround is to avoid symlinks by
    having the actual file in `dst`. If it is a new file (`new_blob=True`), we move it to `dst`. If it is not a new file
    (`new_blob=False`), we don't know if the blob file is already referenced elsewhere. To avoid breaking existing
    cache, the file is duplicated on the disk.

    In case symlinks are not supported, a warning message is displayed to the user once when loading `huggingface_hub`.
    The warning message can be disabled with the `DISABLE_SYMLINKS_WARNING` environment variable.
    """
    try:
        os.remove(dst)
    except OSError:
        pass

    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))
    abs_dst_folder = os.path.dirname(abs_dst)

    # Use relative_dst in priority
    try:
        relative_src = os.path.relpath(abs_src, abs_dst_folder)
    except ValueError:
        # Raised on Windows if src and dst are not on the same volume. This is the case when creating a symlink to a
        # local_dir instead of within the cache directory.
        # See https://docs.python.org/3/library/os.path.html#os.path.relpath
        relative_src = None

    try:
        commonpath = os.path.commonpath([abs_src, abs_dst])
        _support_symlinks = are_symlinks_supported(commonpath)
    except ValueError:
        # Raised if src and dst are not on the same volume. Symlinks will still work on Linux/Macos.
        # See https://docs.python.org/3/library/os.path.html#os.path.commonpath
        _support_symlinks = os.name != "nt" and not constants.HF_HUB_DISABLE_SYMLINKS
    except PermissionError:
        # Permission error means src and dst are not in the same volume (e.g. destination path has been provided
        # by the user via `local_dir`. Let's test symlink support there)
        _support_symlinks = are_symlinks_supported(abs_dst_folder)
    except OSError as e:
        # OS error (errno=30) means that the commonpath is readonly on Linux/MacOS.
        if e.errno == errno.EROFS:
            _support_symlinks = are_symlinks_supported(abs_dst_folder)
        else:
            raise

    # Symlinks are supported => let's create a symlink.
    if _support_symlinks:
        src_rel_or_abs = relative_src or abs_src
        logger.debug(f"Creating pointer from {src_rel_or_abs} to {abs_dst}")
        try:
            os.symlink(src_rel_or_abs, abs_dst)
            return
        except FileExistsError:
            if os.path.islink(abs_dst) and os.path.realpath(abs_dst) == os.path.realpath(abs_src):
                # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
                # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
                return
            else:
                # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
                # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
                raise
        except PermissionError:
            # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
            # is supported on both volumes but not between them. Let's just make a hard copy in that case.
            pass

    # Symlinks are not supported => let's move or copy the file.
    if new_blob:
        logger.debug(f"Symlink not supported. Moving file from {abs_src} to {abs_dst}")
        shutil.move(abs_src, abs_dst, copy_function=_copy_no_matter_what)
    else:
        logger.debug(f"Symlink not supported. Copying file from {abs_src} to {abs_dst}")
        shutil.copyfile(abs_src, abs_dst)


def _cache_commit_hash_for_specific_revision(storage_folder: str, revision: str, commit_hash: str) -> None:
    """Cache reference between a revision (tag, branch or truncated commit hash) and the corresponding commit hash.

    Does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
    """
    if revision != commit_hash:
        ref_path = Path(storage_folder) / "refs" / revision
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        if not ref_path.exists() or commit_hash != ref_path.read_text():
            # Update ref only if has been updated. Could cause useless error in case
            # repo is already cached and user doesn't have write access to cache folder.
            # See https://github.com/huggingface/huggingface_hub/issues/1216.
            # Write atomically (tmp file + rename) so that concurrent readers never see
            # a partially written ref.
            tmp_path = ref_path.with_name(f"{ref_path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp_path.write_text(commit_hash)
            os.replace(tmp_path, ref_path)


@validate_hf_hub_args
def repo_folder_name(*, repo_id: str, repo_type: str) -> str:
    """Return a serialized version of a hf.co repo name and type, safe for disk storage
    as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    # remove all `/` occurrences to correctly convert repo to directory name
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return constants.REPO_ID_SEPARATOR.join(parts)


def _check_disk_space(expected_size: int, target_dir: str | Path) -> None:
    """Check disk usage and log a warning if there is not enough disk space to download the file.

    Args:
        expected_size (`int`):
            The expected size of the file in bytes.
        target_dir (`str`):
            The directory where the file will be stored after downloading.
    """

    target_dir = Path(target_dir)  # format as `Path`
    for path in [target_dir] + list(target_dir.parents):  # first check target_dir, then each parents one by one
        try:
            target_dir_free = shutil.disk_usage(path).free
            if target_dir_free < expected_size:
                warnings.warn(
                    "Not enough free disk space to download the file. "
                    f"The expected file size is: {expected_size / 1e6:.2f} MB. "
                    f"The target location {target_dir} only has {target_dir_free / 1e6:.2f} MB free disk space."
                )
            return
        except OSError:  # raise on anything: file does not exist or space disk cannot be checked
            pass


@overload
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: dict | str | None = None,
    force_download: bool = False,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    dry_run: Literal[False] = False,
) -> str: ...


@overload
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: dict | str | None = None,
    force_download: bool = False,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    dry_run: Literal[True] = True,
) -> DryRunFileInfo: ...


@overload
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: dict | str | None = None,
    force_download: bool = False,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    dry_run: bool = False,
) -> str | DryRunFileInfo: ...


@validate_hf_hub_args
@flag_as_download_call
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: dict | str | None = None,
    force_download: bool = False,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    tqdm_class: type[base_tqdm] | None = None,
    dry_run: bool = False,
) -> str | DryRunFileInfo:
    """Download a given file if it's not already present in the local cache.

    The new cache file layout looks like this:
    - The cache directory contains one subfolder per repo_id (namespaced by repo type)
    - inside each repo folder:
        - refs is a list of the latest known revision => commit_hash pairs
        - blobs contains the actual file blobs (identified by their git-sha or sha256, depending on
          whether they're LFS files or not)
        - snapshots contains one subfolder per commit, each "commit" contains the subset of the files
          that have been resolved at that particular commit. Each filename is a symlink to the blob
          at that particular commit.

    ```
    [  96]  .
    └── [ 160]  models--julien-c--EsperBERTo-small
        ├── [ 160]  blobs
        │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
        ├── [  96]  refs
        │   └── [  40]  main
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
            └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
                ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
                └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    ```

    If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this
    option, the `cache_dir` will not be used and a `.cache/huggingface/` folder will be created at the root of `local_dir`
    to store some metadata related to the downloaded files. While this mechanism is not as robust as the main
    cache-system, it's optimized for regularly pulling the latest version of a repository.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        filename (`str`):
            The name of the file in the repo.
        subfolder (`str`, *optional*):
            An optional value corresponding to a folder inside the model repo.
        repo_type (`str`, *optional*):
            Set to `"dataset"`, `"space"` or `"kernel"` if downloading from a dataset, space or kernel repo,
            `None` or `"model"` if downloading from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        local_dir (`str` or `Path`, *optional*):
            If provided, the downloaded file will be placed under this directory.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in
            the local cache.
        etag_timeout (`float`, *optional*, defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up, which is passed to `httpx2.request`.
        token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        headers (`dict`, *optional*):
            Additional headers to be sent with the request.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.
        tqdm_class (`tqdm`, *optional*):
            If provided, overwrites the default behavior for the progress bar. Passed
            argument must inherit from `tqdm.auto.tqdm` or at least mimic its behavior.
            Defaults to the custom HF progress bar that can be disabled by setting
            `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.
        dry_run (`bool`, *optional*, defaults to `False`):
            If `True`, perform a dry run without actually downloading the file. Returns a
            [`DryRunFileInfo`] object containing information about what would be downloaded.

    Returns:
        `str` or [`DryRunFileInfo`]:
            - If `dry_run=False`: Local path of file or if networking is off, last version of file cached on disk.
            - If `dry_run=True`: A [`DryRunFileInfo`] object containing download information.

    Raises:
        [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist,
            or because it is set to `private` and you do not have access.
        [`~utils.RevisionNotFoundError`]
            If the revision to download from cannot be found.
        [`~utils.RemoteEntryNotFoundError`]
            If the file to download cannot be found.
        [`~utils.LocalEntryNotFoundError`]
            If network is disabled or unavailable and file is not found in cache.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `token=True` but the token cannot be found.
        [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
            If ETag cannot be determined.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If some parameter value is invalid.

    """
    if constants.HF_HUB_ETAG_TIMEOUT != constants.DEFAULT_ETAG_TIMEOUT:
        # Respect environment variable above user value
        etag_timeout = constants.HF_HUB_ETAG_TIMEOUT

    if revision is None:
        revision = constants.DEFAULT_REVISION
    elif isinstance(revision, ResolvedRevision):
        # Revision has already been resolved to a commit hash (see [`HfApi.resolve_revision`]) => use it directly.
        # This pins the download to an immutable commit and lets us skip network calls when it's already cached.
        revision = revision.resolved

    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())

    if local_dir is not None:
        local_dir = str(Path(local_dir).expanduser().resolve())

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = "model"
    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES_WITH_KERNEL)}"
        )

    hf_headers = build_hf_headers(
        token=token,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        headers=headers,
    )

    if local_dir is not None:
        return _hf_hub_download_to_local_dir(
            # Destination
            local_dir=local_dir,
            # File info
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            # HTTP info
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=hf_headers,
            token=token,
            # Additional options
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            tqdm_class=tqdm_class,
            dry_run=dry_run,
        )
    else:
        return _hf_hub_download_to_cache_dir(
            # Destination
            cache_dir=cache_dir,
            # File info
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            # HTTP info
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=hf_headers,
            token=token,
            # Additional options
            local_files_only=local_files_only,
            force_download=force_download,
            tqdm_class=tqdm_class,
            dry_run=dry_run,
        )


def _hf_hub_download_to_cache_dir(
    *,
    # Destination
    cache_dir: str,
    # File info
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    # HTTP info
    endpoint: str | None,
    etag_timeout: float,
    headers: dict[str, str],
    token: bool | str | None,
    # Additional options
    local_files_only: bool,
    force_download: bool,
    tqdm_class: type[base_tqdm] | None,
    dry_run: bool,
) -> str | DryRunFileInfo:
    """Download a given file to a cache folder, if not already present.

    Method should not be called directly. Please use `hf_hub_download` instead.
    """
    locks_dir = os.path.join(cache_dir, ".locks")
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    _validate_relative_filename(filename)
    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))

    # if user provides a commit_hash and they already have the file on disk, shortcut everything.
    if REGEX_COMMIT_HASH.fullmatch(revision):
        pointer_path = _get_pointer_path(storage_folder, revision, relative_filename)
        if os.path.exists(pointer_path):
            if dry_run:
                return DryRunFileInfo(
                    commit_hash=revision,
                    file_size=os.path.getsize(pointer_path),
                    filename=filename,
                    is_cached=True,
                    local_path=pointer_path,
                    will_download=force_download,
                )
            if not force_download:
                return pointer_path

    # Try to get metadata (etag, commit_hash, url, size) from the server.
    # If we can't, a HEAD request error is returned.
    (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = _get_metadata_or_catch_error(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        etag_timeout=etag_timeout,
        headers=headers,
        token=token,
        local_files_only=local_files_only,
        storage_folder=storage_folder,
        relative_filename=relative_filename,
        tree_cache_folder=storage_folder,
    )

    # etag can be None for several reasons:
    # 1. we passed local_files_only.
    # 2. we don't have a connection
    # 3. Hub is down (HTTP 500, 503, 504)
    # 4. repo is not found -for example private or gated- and invalid/missing token sent
    # 5. Hub is blocked by a firewall or proxy is not set correctly.
    # => Try to get the last downloaded one from the specified revision.
    #
    # If the specified revision is a commit hash, look inside "snapshots".
    # If the specified revision is a branch or tag, look inside "refs".
    if head_call_error is not None:
        # Couldn't make a HEAD call => let's try to find a local file
        if not force_download:
            commit_hash = None
            if REGEX_COMMIT_HASH.fullmatch(revision):
                commit_hash = revision
            else:
                ref_path = os.path.join(storage_folder, "refs", revision)
                if os.path.isfile(ref_path):
                    with open(ref_path) as f:
                        commit_hash = f.read()

            # Return pointer file if exists
            if commit_hash is not None:
                pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
                if os.path.exists(pointer_path):
                    if dry_run:
                        return DryRunFileInfo(
                            commit_hash=commit_hash,
                            file_size=os.path.getsize(pointer_path),
                            filename=filename,
                            is_cached=True,
                            local_path=pointer_path,
                            will_download=force_download,
                        )
                    if not force_download:
                        return pointer_path

            if isinstance(head_call_error, _DEFAULT_RETRY_ON_EXCEPTIONS) or (
                isinstance(head_call_error, HfHubHTTPError)
                and head_call_error.response.status_code in _DEFAULT_RETRY_ON_STATUS_CODES
            ):
                logger.info("No local file found. Retrying..")
                (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = (
                    _get_metadata_or_catch_error(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=repo_type,
                        revision=revision,
                        endpoint=endpoint,
                        etag_timeout=_ETAG_RETRY_TIMEOUT,
                        headers=headers,
                        token=token,
                        local_files_only=local_files_only,
                        storage_folder=storage_folder,
                        relative_filename=relative_filename,
                        retry_on_errors=True,
                    )
                )

        # If still error, raise
        if head_call_error is not None:
            _raise_on_head_call_error(head_call_error, force_download, local_files_only)

    # From now on, etag, commit_hash and url are not None.
    assert etag is not None, "etag must have been retrieved from server"
    assert commit_hash is not None, "commit_hash must have been retrieved from server"
    assert url_to_download is not None, "file location must have been retrieved from server"
    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)

    if dry_run:
        # A usable shared store entry counts as cached: the real call would symlink it without downloading.
        # Symlink support is not probed here (probing writes to the cache), so the preview may be optimistic.
        is_cached = (
            os.path.exists(pointer_path)
            or os.path.exists(blob_path)
            or (
                xet_file_data is not None
                and not constants.HF_HUB_DISABLE_SYMLINKS
                and has_shared_blob(xet_hash=xet_file_data.file_hash, cache_dir=cache_dir, expected_size=expected_size)
            )
        )
        return DryRunFileInfo(
            commit_hash=commit_hash,
            file_size=expected_size,
            filename=filename,
            is_cached=is_cached,
            local_path=pointer_path,
            will_download=force_download or not is_cached,
        )

    # Pointer already exists -> update the ref best-effort, then return without
    # attempting to write to the cache (which may be mounted read-only).
    if not force_download and os.path.exists(pointer_path):
        try:
            _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)
        except OSError:
            pass
        return pointer_path

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)

    # Tag cache_dir so backup tools can skip it (CACHEDIR.TAG standard).
    _create_cachedir_tag(Path(cache_dir))

    # if passed revision is not identical to commit_hash
    # then revision has to be a branch name or tag name.
    # In that case store a ref.
    _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)

    # Prevent parallel downloads of the same file with a lock.
    # etag could be duplicated across repos.
    # Note: the lock is best-effort to avoid downloading the same file twice. Cache correctness
    # does not depend on it: each download writes to a process-unique temporary file that is
    # atomically renamed into place (see `_download_to_tmp_and_move`).
    lock_path = os.path.join(locks_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type), f"{etag}.lock")

    lock_path = as_extended_path(lock_path)
    blob_path = as_extended_path(blob_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    # Blob exists but pointer must be (safely) created -> take the lock
    if not force_download and os.path.exists(blob_path):
        with WeakFileLock(lock_path):
            if not os.path.exists(pointer_path):
                _create_symlink(blob_path, pointer_path, new_blob=False)
            return pointer_path

    # Local file doesn't exist or etag isn't a match => retrieve file from remote (or cache)

    # Xet hash of the blob in the shared store, None when the store must not be used for this download
    # (see `utils/_shared_blobs.py`).
    shared_blob_hash = (
        xet_file_data.file_hash
        if xet_file_data is not None and shared_blobs_enabled() and are_symlinks_supported(cache_dir)
        else None
    )

    with WeakFileLock(lock_path):
        blob_reused_from_store = (
            shared_blob_hash is not None
            and not force_download
            and not os.path.exists(blob_path)
            and try_link_from_shared_store(
                blob_path=blob_path, xet_hash=shared_blob_hash, cache_dir=cache_dir, expected_size=expected_size
            )
        )
        blob_is_shared = blob_reused_from_store
        if not blob_reused_from_store:
            will_download = force_download or not os.path.exists(blob_path)
            _download_to_tmp_and_move(
                incomplete_path=Path(blob_path + ".incomplete"),
                destination_path=Path(blob_path),
                url_to_download=url_to_download,
                headers=headers,
                expected_size=expected_size,
                filename=filename,
                force_download=force_download,
                etag=etag,
                xet_file_data=xet_file_data,
                tqdm_class=tqdm_class,
            )
            if shared_blob_hash is not None and will_download and is_xet_available():
                # Only Xet-verified downloads are published, never the plain HTTP fallback.
                blob_is_shared = publish_blob_to_shared_store(
                    blob_path=blob_path,
                    xet_hash=shared_blob_hash,
                    cache_dir=cache_dir,
                    expected_size=expected_size,
                    replace_existing=force_download,
                )
        if not os.path.exists(pointer_path):
            # A shared blob is a symlink: if the snapshot symlink fails, copy through it instead of moving it away.
            _create_symlink(blob_path, pointer_path, new_blob=not blob_is_shared)

    return pointer_path


def _hf_hub_download_to_local_dir(
    *,
    # Destination
    local_dir: str | Path,
    # File info
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str,
    # HTTP info
    endpoint: str | None,
    etag_timeout: float,
    headers: dict[str, str],
    token: bool | str | None,
    # Additional options
    cache_dir: str,
    force_download: bool,
    local_files_only: bool,
    tqdm_class: type[base_tqdm] | None,
    dry_run: bool,
) -> str | DryRunFileInfo:
    """Download a given file to a local folder, if not already present.

    Method should not be called directly. Please use `hf_hub_download` instead.
    """
    local_dir = Path(as_extended_path(local_dir))
    paths = get_local_download_paths(local_dir=local_dir, filename=filename)
    local_metadata = read_download_metadata(local_dir=local_dir, filename=filename)

    # Local file exists + metadata exists + commit_hash matches => return file
    if (
        REGEX_COMMIT_HASH.fullmatch(revision)
        and paths.file_path.is_file()
        and local_metadata is not None
        and local_metadata.commit_hash == revision
    ):
        local_file = str(paths.file_path)
        if dry_run:
            return DryRunFileInfo(
                commit_hash=revision,
                file_size=os.path.getsize(local_file),
                filename=filename,
                is_cached=True,
                local_path=local_file,
                will_download=force_download,
            )
        if not force_download:
            return local_file

    # Local file doesn't exist or commit_hash doesn't match => we need the etag
    # - Xet file => might be cached in /tree listing
    # - otherwise => HEAD /resolve call
    tree_cache_folder = tree_cache_folder_for_local_dir(str(local_dir))
    (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = _get_metadata_or_catch_error(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        etag_timeout=etag_timeout,
        headers=headers,
        token=token,
        local_files_only=local_files_only,
        tree_cache_folder=tree_cache_folder,
    )

    if head_call_error is not None:
        # No HEAD call but local file exists => default to local file
        if paths.file_path.is_file():
            if dry_run or not force_download:
                logger.warning(
                    f"Couldn't access the Hub to check for update but local file already exists. Defaulting to existing file. (error: {head_call_error})"
                )
            local_path = str(paths.file_path)
            if dry_run and local_metadata is not None:
                return DryRunFileInfo(
                    commit_hash=local_metadata.commit_hash,
                    file_size=os.path.getsize(local_path),
                    filename=filename,
                    is_cached=True,
                    local_path=local_path,
                    will_download=force_download,
                )
            if not force_download:
                return local_path
        elif not force_download:
            if isinstance(head_call_error, _DEFAULT_RETRY_ON_EXCEPTIONS) or (
                isinstance(head_call_error, HfHubHTTPError)
                and head_call_error.response.status_code in _DEFAULT_RETRY_ON_STATUS_CODES
            ):
                logger.info("No local file found. Retrying..")
                (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = (
                    _get_metadata_or_catch_error(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=repo_type,
                        revision=revision,
                        endpoint=endpoint,
                        etag_timeout=_ETAG_RETRY_TIMEOUT,
                        headers=headers,
                        token=token,
                        local_files_only=local_files_only,
                        retry_on_errors=True,
                    )
                )

        # If still error, raise
        if head_call_error is not None:
            _raise_on_head_call_error(head_call_error, force_download, local_files_only)

    # From now on, etag, commit_hash and url are not None.
    assert etag is not None, "etag must have been retrieved from server"
    assert commit_hash is not None, "commit_hash must have been retrieved from server"
    assert url_to_download is not None, "file location must have been retrieved from server"

    # Local file exists => check if it's up-to-date
    if not force_download and paths.file_path.is_file():
        # etag matches => update metadata and return file
        if local_metadata is not None and local_metadata.etag == etag:
            write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
            if dry_run:
                return DryRunFileInfo(
                    commit_hash=commit_hash,
                    file_size=expected_size,
                    filename=filename,
                    is_cached=True,
                    local_path=str(paths.file_path),
                    will_download=False,
                )
            return str(paths.file_path)

        # metadata is outdated + etag is a sha256
        # => means it's an LFS file (large)
        # => let's compute local hash and compare
        # => if match, update metadata and return file
        if local_metadata is None and REGEX_SHA256.fullmatch(etag) is not None:
            with open(paths.file_path, "rb") as f:
                file_hash = sha_fileobj(f).hex()
            if file_hash == etag:
                write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
                if dry_run:
                    return DryRunFileInfo(
                        commit_hash=commit_hash,
                        file_size=expected_size,
                        filename=filename,
                        is_cached=True,
                        local_path=str(paths.file_path),
                        will_download=False,
                    )
                return str(paths.file_path)

    # Local file doesn't exist or etag isn't a match => retrieve file from remote (or cache)

    # If we are lucky enough, the file is already in the cache => copy it
    if not force_download:
        cached_path = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=commit_hash,
            repo_type=repo_type,
        )
        if isinstance(cached_path, str):
            if dry_run:
                return DryRunFileInfo(
                    commit_hash=commit_hash,
