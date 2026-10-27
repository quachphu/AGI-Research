                    file_size=expected_size,
                    filename=filename,
                    is_cached=True,
                    local_path=str(paths.file_path),
                    will_download=False,
                )
            with WeakFileLock(paths.lock_path):
                paths.file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_path, paths.file_path)
            write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
            return str(paths.file_path)

    if dry_run:
        is_cached = paths.file_path.is_file()
        return DryRunFileInfo(
            commit_hash=commit_hash,
            file_size=expected_size,
            filename=filename,
            is_cached=is_cached,
            local_path=str(paths.file_path),
            will_download=force_download or not is_cached,
        )

    # Otherwise, let's download the file!
    with WeakFileLock(paths.lock_path):
        paths.file_path.unlink(missing_ok=True)  # delete outdated file first
        _download_to_tmp_and_move(
            incomplete_path=paths.incomplete_path(etag),
            destination_path=paths.file_path,
            url_to_download=url_to_download,
            headers=headers,
            expected_size=expected_size,
            filename=filename,
            force_download=force_download,
            etag=etag,
            xet_file_data=xet_file_data,
            tqdm_class=tqdm_class,
        )

    write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
    return str(paths.file_path)


@validate_hf_hub_args
def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    repo_type: str | None = None,
) -> str | _CACHED_NO_EXIST_T | None:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repository. Will default to `"model"`.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.

    Example:

    ```python
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    filepath = try_to_load_from_cache()
    if isinstance(filepath, str):
        # file exists and is cached
        ...
    elif filepath is _CACHED_NO_EXIST:
        # non-existence of file is cached
        ...
    else:
        # file is not cached
        ...
    ```
    """
    if revision is None:
        revision = "main"
    if repo_type is None:
        repo_type = "model"
    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES_WITH_KERNEL)}"
        )
    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE

    repo_cache = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    no_exist_dir = os.path.join(repo_cache, ".no_exist")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if file is cached as "no_exist"
    if os.path.isfile(os.path.join(no_exist_dir, revision, filename)):
        return _CACHED_NO_EXIST

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = os.path.join(snapshots_dir, revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


@validate_hf_hub_args
def get_hf_file_metadata(
    url: str,
    token: bool | str | None = None,
    timeout: float | None = constants.HF_HUB_ETAG_TIMEOUT,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    retry_on_errors: bool = False,
) -> HfFileMetadata:
    """Fetch metadata of a file versioned on the Hub for a given url.

    Args:
        url (`str`):
            File url, for example returned by [`hf_hub_url`].
        token (`str` or `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If `False` or `None`, no token is provided.
                - If a string, it's used as the authentication token.
        timeout (`float`, *optional*, defaults to 10):
            How many seconds to wait for the server to send metadata before giving up.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        headers (`dict`, *optional*):
            Additional headers to be sent with the request.
        endpoint (`str`, *optional*):
            Endpoint of the Hub. Defaults to <https://huggingface.co>.
        retry_on_errors (`bool`, *optional*, defaults to `False`):
            Whether to retry on errors (429, 5xx, timeout, network errors).
            If False, no retry for fast fallback to local cache.

    Returns:
        A [`HfFileMetadata`] object containing metadata such as location, etag, size and
        commit_hash.
    """
    hf_headers = build_hf_headers(
        token=token,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        headers=headers,
    )
    hf_headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

    # Retrieve metadata
    response = _httpx2_follow_hub_redirects_with_backoff(
        method="HEAD", url=url, headers=hf_headers, timeout=timeout, retry_on_errors=retry_on_errors
    )
    hf_raise_for_status(response)

    # Return
    return HfFileMetadata(
        commit_hash=response.headers.get(constants.HUGGINGFACE_HEADER_X_REPO_COMMIT),
        # We favor a custom header indicating the etag of the linked resource, and we fall back to the regular etag header.
        etag=_normalize_etag(
            response.headers.get(constants.HUGGINGFACE_HEADER_X_LINKED_ETAG) or response.headers.get("ETag")
        ),
        # Either from response headers (if redirected) or defaults to request url
        # Do not use directly `url` as we might have followed relative redirects.
        location=response.headers.get("Location") or str(response.request.url),  # type: ignore
        size=_int_or_none(
            response.headers.get(constants.HUGGINGFACE_HEADER_X_LINKED_SIZE)
            or (None if response.is_redirect else response.headers.get("Content-Length"))
        ),
        xet_file_data=parse_xet_file_data_from_response(response, endpoint=endpoint),  # type: ignore
    )


def _get_metadata_or_catch_error(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    endpoint: str | None,
    etag_timeout: float | None,
    headers: dict[str, str],  # mutated inplace!
    token: bool | str | None,
    local_files_only: bool,
    relative_filename: str | None = None,  # only used to store `.no_exists` in cache
    storage_folder: str | None = None,  # only used to store `.no_exists` in cache
    retry_on_errors: bool = False,
    tree_cache_folder: str | None = None,  # if set, read the on-disk tree listing to skip the HEAD call
) -> (
    # Either an exception is caught and returned
    tuple[None, None, None, None, None, Exception]
    |
    # Or the metadata is returned as
    # `(url_to_download, etag, commit_hash, expected_size, xet_file_data, None)`
    tuple[str, str, str, int | None, XetFileData | None, None]
):
    """Get metadata for a file on the Hub, safely handling network issues.

    Returns either the etag, commit_hash and expected size of the file, or the error
    raised while fetching the metadata.

    NOTE: This function mutates `headers` inplace! It removes the `authorization` header
          if the file is a LFS blob and the domain of the url is different from the
          domain of the location (typically an S3 bucket).
    """
    if local_files_only:
        return (
            None,
            None,
            None,
            None,
            None,
            OfflineModeIsEnabled(
                f"Cannot access file since 'local_files_only=True' as been set. (repo_id: {repo_id}, repo_type: {repo_type}, revision: {revision}, filename: {filename})"
            ),
        )

    # Skip the per-file HEAD call when the file metadata can be rebuilt from a tree listing cached on disk.
    if tree_cache_folder is not None and REGEX_COMMIT_HASH.fullmatch(revision):
        tree_metadata = _xet_file_metadata_from_tree_cache(
            tree_cache_folder=tree_cache_folder,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_hash=revision,
            filename=filename,
            endpoint=endpoint,
        )
        if tree_metadata is not None:
            return tree_metadata

    url = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)
    url_to_download: str = url
    etag: str | None = None
    commit_hash: str | None = None
    expected_size: int | None = None
    head_error_call: Exception | None = None
    xet_file_data: XetFileData | None = None

    # Try to get metadata from the server.
    # Do not raise yet if the file is not found or not accessible.
    if not local_files_only:
        try:
            try:
                metadata = get_hf_file_metadata(
                    url=url,
                    timeout=etag_timeout,
                    headers=headers,
                    token=token,
                    endpoint=endpoint,
                    retry_on_errors=retry_on_errors,
                )
            except RemoteEntryNotFoundError as http_error:
                if storage_folder is not None and relative_filename is not None:
                    # Cache the non-existence of the file
                    commit_hash = http_error.response.headers.get(constants.HUGGINGFACE_HEADER_X_REPO_COMMIT)
                    if commit_hash is not None:
                        no_exist_file_path = Path(storage_folder) / ".no_exist" / commit_hash / relative_filename
                        if not no_exist_file_path.exists():
                            try:
                                no_exist_file_path.parent.mkdir(parents=True, exist_ok=True)
                                no_exist_file_path.touch()
                            except OSError as e:
                                logger.error(
                                    f"Could not cache non-existence of file. Will ignore error and continue. Error: {e}"
                                )
                        _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)
                raise

            # Commit hash must exist
            commit_hash = metadata.commit_hash
            if commit_hash is None:
                raise FileMetadataError(
                    f"Response from {url} is missing the '{constants.HUGGINGFACE_HEADER_X_REPO_COMMIT}' header, so it"
                    " does not seem to be served by a Hugging Face Hub endpoint. If HF_ENDPOINT is set, check that it"
                    " points to a Hub-compatible endpoint. Otherwise, check your firewall and proxy settings and make"
                    " sure your SSL certificates are updated."
                )

            # Etag must exist
            # If we don't have any of those, raise an error.
            etag = metadata.etag
            if etag is None:
                raise FileMetadataError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )

            # Xet downloads require a known size, but regular HTTP downloads can recover it from the GET response.
            expected_size = metadata.size
            xet_file_data = metadata.xet_file_data
            if expected_size is None and xet_file_data is not None and is_xet_available():
                raise FileMetadataError("Distant resource does not have a Content-Length.")

            # In case of a redirect, save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            #
            # If the final location is on a Hub host (same host, or e.g. huggingface.co reached through an
            # HF_ENDPOINT mirror redirect) => keep auth. Otherwise it's a signed CDN url => don't send auth.
            if xet_file_data is None and url != metadata.location:
                url_to_download = metadata.location
                if not _is_same_or_hub_host(url, metadata.location):
                    # Remove authorization header when downloading a LFS blob from a CDN
                    headers.pop("authorization", None)
        except httpx2.ProxyError:
            # Actually raise on proxy error
            raise
        except (httpx2.ConnectError, httpx2.TimeoutException, OfflineModeIsEnabled) as error:
            # Otherwise, our Internet connection is down.
            # etag is None
            head_error_call = error
        except (RevisionNotFoundError, RemoteEntryNotFoundError):
            # The repo was found but the revision or entry doesn't exist on the Hub (never existed or got deleted)
            raise
        except HfHubHTTPError as error:
            # Multiple reasons for an http error:
            # - Repository is private and invalid/missing token sent
            # - Repository is gated and invalid/missing token sent
            # - Hub is down (error 500 or 504)
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_error_call = error
        except FileMetadataError as error:
            # Multiple reasons for a FileMetadataError:
            # - Wrong network configuration (proxy, firewall, SSL certificates)
            # - Inconsistency on the Hub
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_error_call = error

    if not (local_files_only or etag is not None or head_error_call is not None):
        raise RuntimeError("etag is empty due to uncovered problems")

    if head_error_call is not None:
        _detach_tracebacks(head_error_call)

    return (url_to_download, etag, commit_hash, expected_size, xet_file_data, head_error_call)  # type: ignore


def _detach_tracebacks(error: BaseException) -> None:
    """Detach the tracebacks of an exception and of its whole `__cause__`/`__context__` chain.

    Returning an exception instead of raising it means keeping its traceback alive. A traceback references the frames
    it was raised from, and each frame references its caller (up to the user code calling `hf_hub_download`) and its
    own local variables - including the returned exception itself. This creates a reference cycle that only the cyclic
    garbage collector can break, so callers falling back to a cached file (i.e. discarding the error) keep their own
    stack - and everything it references - alive until the next `gc.collect()`.

    Only the frames are dropped: exception types, messages and the cause/context chain are preserved, so re-raising a
    detached error still reports what went wrong.

    See https://github.com/vllm-project/vllm/pull/50341 for a real-world report of this retention.
    """
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:  # defensive: exception chains can be cyclic
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend(exc for exc in (current.__cause__, current.__context__) if exc is not None)


def _xet_file_metadata_from_tree_cache(
    *,
    tree_cache_folder: str,
    repo_id: str,
    repo_type: str,
    commit_hash: str,
    filename: str,
    endpoint: str | None,
) -> tuple[str, str, str, int, XetFileData, None] | None:
    """Rebuild the metadata a HEAD call would return, from the on-disk tree listing cache.

    The optimization is intentionally limited to Xet files when Xet is enabled: that's the only case where
    skipping the HEAD call pays off, since Xet downloads don't rely on the `/resolve` redirect the HEAD call
    would resolve. For regular files (or when Xet is unavailable) we return `None` and let the caller make the
    HEAD call as usual. Also returns `None` when the commit's tree listing or the file is not cached.
    """
    if not is_xet_available():
        return None
    tree_entries = read_tree_cache(tree_cache_folder, commit_hash)
    if tree_entries is None:
        return None
    entry = tree_entries.get(filename)
    if (
        entry is None
        or entry.xet_hash is None
        or not is_valid_xet_hash(entry.xet_hash)
        or entry.lfs_sha256 is None
        or entry.lfs_size is None
    ):
        return None

    xet_file_data = XetFileData(
        file_hash=entry.xet_hash,
        refresh_route=xet_connection_info_refresh_url(
            token_type=XetTokenType.READ,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=commit_hash,
            endpoint=endpoint,
        ),
    )
    # Mirror the server's HEAD response: ETag/size come from the LFS metadata for LFS-tracked files (which Xet
    # files are).
    etag = entry.lfs_sha256
    file_size = entry.lfs_size
    location = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=commit_hash, endpoint=endpoint)
    return (location, etag, commit_hash, file_size, xet_file_data, None)


def _raise_on_head_call_error(head_call_error: Exception, force_download: bool, local_files_only: bool) -> NoReturn:
    """Raise an appropriate error when the HEAD call failed and we cannot locate a local file."""
    # No head call => we cannot force download.
    if force_download:
        if local_files_only:
            raise ValueError("Cannot pass 'force_download=True' and 'local_files_only=True' at the same time.")
        elif isinstance(head_call_error, OfflineModeIsEnabled):
            raise ValueError("Cannot pass 'force_download=True' when offline mode is enabled.") from head_call_error
        else:
            raise ValueError("Force download failed due to the above error.") from head_call_error

    # No head call + couldn't find an appropriate file on disk => raise an error.
    if local_files_only:
        raise LocalEntryNotFoundError(
            "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"
            " hf.co look-ups and downloads online, set 'local_files_only' to False."
        )
    elif isinstance(head_call_error, (RepositoryNotFoundError, GatedRepoError)) or (
        isinstance(head_call_error, HfHubHTTPError) and head_call_error.response.status_code == 401
    ):
        # Repo not found or gated => let's raise the actual error
        # Unauthorized => likely a token issue => let's raise the actual error
        raise head_call_error
    elif isinstance(head_call_error, FileMetadataError):
        # The call succeeded but the response lacked the metadata we need => a configuration issue, not connectivity.
        raise LocalEntryNotFoundError(
            f"{head_call_error} We also cannot find the requested files in the local cache."
        ) from head_call_error
    else:
        # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
        raise LocalEntryNotFoundError(
            "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
            " in the local cache. Please check your connection and try again or make sure your Internet connection"
            " is on."
        ) from head_call_error


def _download_to_tmp_and_move(
    incomplete_path: Path,
    destination_path: Path,
    url_to_download: str,
    headers: dict[str, str],
    expected_size: int | None,
    filename: str,
    force_download: bool,
    etag: str | None,
    xet_file_data: XetFileData | None,
    tqdm_class: type[base_tqdm] | None = None,
) -> None:
    """Download content from a URL to a destination path.

    Internal logic:
    - return early if file is already downloaded
    - check disk space before downloading
    - download content to a process-unique temporary file
    - set correct permissions on temporary file
    - move the temporary file to the destination path

    Both `incomplete_path` and `destination_path` must be on the same volume to avoid a local copy.
    """
    if destination_path.exists() and not force_download:
        # Do nothing if already exists (except if force_download=True)
        return

    # Download to a process-unique temporary file before moving it in place. A shared
    # `<etag>.incomplete` file corrupts the cache whenever the surrounding lock is not honored:
    # on some filesystems (Lustre, GPFS, some NFS mounts) `flock(2)` silently succeeds for every
    # caller and concurrent processes end up appending to the same file. With a unique file per
    # process, a broken lock costs only duplicated bandwidth: each process downloads the full
    # file and atomically renames it to the final destination.
    # See https://github.com/huggingface/huggingface_hub/pull/4228.
    # Re-check the length on Windows: the suffix can push a path just under the limit above MAX_PATH.
    tmp_path = incomplete_path.with_name(f"{incomplete_path.stem}.{uuid.uuid4().hex[:8]}.incomplete")
    tmp_path = Path(as_extended_path(tmp_path))
    try:
        with tmp_path.open("wb") as f:
            logger.debug(f"Downloading '{filename}' to '{tmp_path}'")

            if expected_size is not None:  # might be None if HTTP header not set correctly
                # Check disk space in both tmp and destination path
                _check_disk_space(expected_size, tmp_path.parent)
                _check_disk_space(expected_size, destination_path.parent)

            if xet_file_data is not None and is_xet_available():
                logger.debug("Xet Storage is enabled for this repo. Downloading file from Xet Storage..")
                xet_get(
                    incomplete_path=tmp_path,
                    xet_file_data=xet_file_data,
                    headers=headers,
                    expected_size=expected_size,
                    displayed_filename=filename,
                    tqdm_class=tqdm_class,
                )
            else:
                if xet_file_data is not None and not constants.HF_HUB_DISABLE_XET:
                    logger.warning(
                        "Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. "
                        "Falling back to regular HTTP download. "
                        "For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`"
                    )

                http_get(
                    url_to_download,
                    f,
                    headers=headers,
                    expected_size=expected_size,
                    tqdm_class=tqdm_class,
                )

        logger.debug(f"Download complete. Moving file to {destination_path}")
        _chmod_and_move(tmp_path, destination_path)
    finally:
        # No-op on success (file has been moved). On failure, do not keep a partial file around:
        # it could not be reused anyway since the temporary name is unique to this download.
        tmp_path.unlink(missing_ok=True)


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value)  # type: ignore
    except (TypeError, ValueError):
        return None


def _chmod_and_move(src: Path, dst: Path) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    # Get umask by creating a temporary file in the folder containing the incomplete file.
    # We know this folder is writable since the incomplete file has just been written there.
    # Probing next to `dst` is not always possible: when downloading to a local dir, `dst` is the
    # final file location and its parents might not be writable (e.g. read-only root filesystem).
    # See https://github.com/huggingface/huggingface_hub/issues/4304.
    tmp_file = src.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(str(src), stat.S_IMODE(cache_dir_mode))
    except OSError as e:
        logger.warning(
            f"Could not set the permissions on the file '{src}'. Error: {e}.\nContinuing without setting permissions."
        )
    finally:
        try:
            tmp_file.unlink()
        except OSError:
            # fails if `tmp_file.touch()` failed => do nothing
            # See https://github.com/huggingface/huggingface_hub/issues/2359
            pass

    if os.path.lexists(dst):
        # Replace the entry so a force download never writes through a shared symlink.
        _replace_no_matter_what(src, dst)
    else:
        shutil.move(str(src), str(dst), copy_function=_copy_no_matter_what)


def _replace_no_matter_what(src: Path, dst: Path) -> None:
    """Replace `dst` with `src`.

    Some mounts reject replace-over-existing: stage the new file next to `dst`, move the old entry aside
    and restore it if the final move fails.
    """
    try:
        os.replace(src, dst)
    except OSError:
        staged_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.new")
        backup_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.old")
        backup_holds_previous_entry = False
        try:
            shutil.move(str(src), str(staged_dst), copy_function=_copy_no_matter_what)
            os.rename(dst, backup_dst)
            backup_holds_previous_entry = True
            try:
                shutil.move(str(staged_dst), str(dst), copy_function=_copy_no_matter_what)
            except OSError as move_error:
                try:
                    if os.path.lexists(dst):
                        os.unlink(dst)
                    os.rename(backup_dst, dst)
                    backup_holds_previous_entry = False
                except OSError as restore_error:
                    raise OSError(
                        f"Could not restore previous destination '{dst}' from '{backup_dst}'"
                    ) from restore_error
                raise move_error
            try:
                backup_dst.unlink()
                backup_holds_previous_entry = False
            except OSError as cleanup_error:
                logger.warning(f"Could not remove previous destination backup '{backup_dst}': {cleanup_error}")
        finally:
            staged_dst.unlink(missing_ok=True)
            if not backup_holds_previous_entry:
                backup_dst.unlink(missing_ok=True)


def _copy_no_matter_what(src: str, dst: str) -> None:
    """Copy file from src to dst.

    If `shutil.copy2` fails, fallback to `shutil.copyfile`.
    """
    try:
        # Copy file with metadata and permission
        # Can fail e.g. if dst is an S3 mount
        shutil.copy2(src, dst)
    except OSError:
        # Copy only file content
        shutil.copyfile(src, dst)


def _get_pointer_path(storage_folder: str, revision: str, relative_filename: str) -> str:
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    snapshot_path = os.path.join(storage_folder, "snapshots")
    pointer_path = os.path.join(snapshot_path, revision, relative_filename)
    if Path(os.path.abspath(snapshot_path)) not in Path(os.path.abspath(pointer_path)).parents:
        raise ValueError(
            "Invalid pointer path: cannot create pointer path in snapshot folder if"
            f" `storage_folder='{storage_folder}'`, `revision='{revision}'` and"
            f" `relative_filename='{relative_filename}'`."
        )
    return pointer_path
<|endoftext|>"""
Data structures to interact with Discussions and Pull Requests on the Hub.

See [the Discussions and Pull Requests guide](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
for more information on Pull Requests, Discussions, and the community tab.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from . import constants
from .utils import parse_datetime


DiscussionStatus = Literal["open", "closed", "merged", "draft"]


@dataclass
class Discussion:
    """
    A Discussion or Pull Request on the Hub.

    This dataclass is not intended to be instantiated directly.

    Attributes:
        title (`str`):
            The title of the Discussion / Pull Request
        status (`str`):
            The status of the Discussion / Pull Request.
            It must be one of:
                * `"open"`
                * `"closed"`
                * `"merged"` (only for Pull Requests )
                * `"draft"` (only for Pull Requests )
        num (`int`):
            The number of the Discussion / Pull Request.
        repo_id (`str`):
            The id (`"{namespace}/{repo_name}"`) of the repo on which
            the Discussion / Pull Request was open.
        repo_type (`str`):
            The type of the repo on which the Discussion / Pull Request was open.
            Possible values are: `"model"`, `"dataset"`, `"space"`.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        is_pull_request (`bool`):
            Whether or not this is a Pull Request.
        created_at (`datetime`):
            The `datetime` of creation of the Discussion / Pull Request.
        endpoint (`str`):
            Endpoint of the Hub. Default is https://huggingface.co.
        git_reference (`str`, *optional*):
            (property) Git reference to which changes can be pushed if this is a Pull Request, `None` otherwise.
        url (`str`):
            (property) URL of the discussion on the Hub.
    """

    title: str
    status: DiscussionStatus
    num: int
    repo_id: str
    repo_type: str
    author: str
    is_pull_request: bool
    created_at: datetime
    endpoint: str

    @property
    def git_reference(self) -> str | None:
        """
        If this is a Pull Request , returns the git reference to which changes can be pushed.
        Returns `None` otherwise.
        """
        if self.is_pull_request:
            return f"refs/pr/{self.num}"
        return None

    @property
    def url(self) -> str:
        """Returns the URL of the discussion on the Hub."""
        if self.repo_type is None or self.repo_type == constants.REPO_TYPE_MODEL:
            return f"{self.endpoint}/{self.repo_id}/discussions/{self.num}"
        return f"{self.endpoint}/{self.repo_type}s/{self.repo_id}/discussions/{self.num}"


@dataclass
class DiscussionWithDetails(Discussion):
    """
    Subclass of [`Discussion`].

    Attributes:
        title (`str`):
            The title of the Discussion / Pull Request
        status (`str`):
            The status of the Discussion / Pull Request.
            It can be one of:
                * `"open"`
                * `"closed"`
                * `"merged"` (only for Pull Requests )
                * `"draft"` (only for Pull Requests )
        num (`int`):
            The number of the Discussion / Pull Request.
        repo_id (`str`):
            The id (`"{namespace}/{repo_name}"`) of the repo on which
            the Discussion / Pull Request was open.
        repo_type (`str`):
            The type of the repo on which the Discussion / Pull Request was open.
            Possible values are: `"model"`, `"dataset"`, `"space"`.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        is_pull_request (`bool`):
            Whether or not this is a Pull Request.
        created_at (`datetime`):
            The `datetime` of creation of the Discussion / Pull Request.
        events (`list` of [`DiscussionEvent`]):
            The list of [`DiscussionEvents`] in this Discussion or Pull Request.
        conflicting_files (`Union[list[str], bool, None]`, *optional*):
            A list of conflicting files if this is a Pull Request.
            `None` if `self.is_pull_request` is `False`.
            `True` if there are conflicting files but the list can't be retrieved.
        target_branch (`str`, *optional*):
            The branch into which changes are to be merged if this is a
            Pull Request . `None`  if `self.is_pull_request` is `False`.
        merge_commit_oid (`str`, *optional*):
            If this is a merged Pull Request , this is set to the OID / SHA of
            the merge commit, `None` otherwise.
        diff (`str`, *optional*):
            The git diff if this is a Pull Request , `None` otherwise.
        endpoint (`str`):
            Endpoint of the Hub. Default is https://huggingface.co.
        git_reference (`str`, *optional*):
            (property) Git reference to which changes can be pushed if this is a Pull Request, `None` otherwise.
        url (`str`):
            (property) URL of the discussion on the Hub.
    """

    events: list["DiscussionEvent"]
    conflicting_files: list[str] | bool | None
    target_branch: str | None
    merge_commit_oid: str | None
    diff: str | None


class DiscussionEventArgs(TypedDict):
    id: str
    type: str
    created_at: datetime
    author: str
    _event: dict


@dataclass
class DiscussionEvent:
    """
    An event in a Discussion or Pull Request.

    Use concrete classes:
        * [`DiscussionComment`]
        * [`DiscussionStatusChange`]
        * [`DiscussionCommit`]
        * [`DiscussionTitleChange`]

    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
    """

    id: str
    type: str
    created_at: datetime
    author: str

    _event: dict
    """Stores the original event data, in case we need to access it later."""


@dataclass
class DiscussionComment(DiscussionEvent):
    """A comment in a Discussion / Pull Request.

    Subclass of [`DiscussionEvent`].


    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        content (`str`):
            The raw markdown content of the comment. Mentions, links and images are not rendered.
        edited (`bool`):
            Whether or not this comment has been edited.
        hidden (`bool`):
            Whether or not this comment has been hidden.
    """

    content: str
    edited: bool
    hidden: bool

    @property
    def rendered(self) -> str:
        """The rendered comment, as a HTML string"""
        return self._event["data"]["latest"]["html"]

    @property
    def last_edited_at(self) -> datetime:
        """The last edit time, as a `datetime` object."""
        return parse_datetime(self._event["data"]["latest"]["updatedAt"])

    @property
    def last_edited_by(self) -> str:
        """The username of the user who last edited the comment."""
        return self._event["data"]["latest"].get("author", {}).get("name", "deleted")

    @property
    def edit_history(self) -> list[dict]:
        """The edit history of the comment"""
        return self._event["data"]["history"]

    @property
    def number_of_edits(self) -> int:
        return len(self.edit_history)


@dataclass
class DiscussionStatusChange(DiscussionEvent):
    """A change of status in a Discussion / Pull Request.

    Subclass of [`DiscussionEvent`].

    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        new_status (`str`):
            The status of the Discussion / Pull Request after the change.
            It can be one of:
                * `"open"`
                * `"closed"`
                * `"merged"` (only for Pull Requests )
    """

    new_status: str


@dataclass
class DiscussionCommit(DiscussionEvent):
    """A commit in a Pull Request.

    Subclass of [`DiscussionEvent`].

    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        summary (`str`):
            The summary of the commit.
        oid (`str`):
            The OID / SHA of the commit, as a hexadecimal string.
    """

    summary: str
    oid: str


@dataclass
class DiscussionTitleChange(DiscussionEvent):
    """A rename event in a Discussion / Pull Request.

    Subclass of [`DiscussionEvent`].

    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        old_title (`str`):
            The previous title for the Discussion / Pull Request.
        new_title (`str`):
            The new title.
    """

    old_title: str
    new_title: str


def deserialize_event(event: dict) -> DiscussionEvent:
    """Instantiates a [`DiscussionEvent`] from a dict"""
    event_id: str = event["id"]
    event_type: str = event["type"]
    created_at = parse_datetime(event["createdAt"])

    common_args: DiscussionEventArgs = {
        "id": event_id,
        "type": event_type,
        "created_at": created_at,
        "author": event.get("author", {}).get("name", "deleted"),
        "_event": event,
    }

    if event_type == "comment":
        return DiscussionComment(
            **common_args,
            edited=event["data"]["edited"],
            hidden=event["data"]["hidden"],
            content=event["data"]["latest"]["raw"],
        )
    if event_type == "status-change":
        return DiscussionStatusChange(
            **common_args,
            new_status=event["data"]["status"],
        )
    if event_type == "commit":
        return DiscussionCommit(
            **common_args,
            summary=event["data"]["subject"],
            oid=event["data"]["oid"],
        )
    if event_type == "title-change":
        return DiscussionTitleChange(
            **common_args,
            old_title=event["data"]["from"],
            new_title=event["data"]["to"],
        )

    return DiscussionEvent(**common_args)
<|endoftext|>import os
import re
import typing
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit


# Possible values for env variables


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
REPOCARD_NAME = "README.md"
EVAL_RESULTS_FOLDER = ".eval_results"
DEFAULT_ETAG_TIMEOUT = 10
DEFAULT_DOWNLOAD_TIMEOUT = 10
DEFAULT_REQUEST_TIMEOUT = 10
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
MAX_HTTP_DOWNLOAD_SIZE = 50 * 1000 * 1000 * 1000  # 50 GB

# Constants for serialization

PYTORCH_WEIGHTS_FILE_PATTERN = "pytorch_model{suffix}.bin"  # Unsafe pickle: use safetensors instead
SAFETENSORS_WEIGHTS_FILE_PATTERN = "model{suffix}.safetensors"
TF2_WEIGHTS_FILE_PATTERN = "tf_model{suffix}.h5"

# Constants for safetensors repos

SAFETENSORS_SINGLE_FILE = "model.safetensors"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_MAX_HEADER_LENGTH = 25_000_000

# Timeout of acquiring file lock and logging the attempt
FILELOCK_LOG_EVERY_SECONDS = 10

# Git-related constants

DEFAULT_REVISION = "main"
REGEX_COMMIT_OID = re.compile(r"[A-Fa-f0-9]{5,40}")

HUGGINGFACE_CO_URL_HOME = "https://huggingface.co/"

_staging_mode = _is_true(os.environ.get("HUGGINGFACE_CO_STAGING"))

_HF_DEFAULT_ENDPOINT = "https://huggingface.co"
_HF_DEFAULT_STAGING_ENDPOINT = "https://hub-ci.huggingface.co"
ENDPOINT = os.getenv("HF_ENDPOINT", _HF_DEFAULT_ENDPOINT).rstrip("/")
HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"

if _staging_mode:
    ENDPOINT = _HF_DEFAULT_STAGING_ENDPOINT
    HUGGINGFACE_CO_URL_TEMPLATE = _HF_DEFAULT_STAGING_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"

# Hosts considered to be Hugging Face Hub endpoints: the public Hub host, its ``hf.co`` short domain, the staging
# host, and the host of the configured ``ENDPOINT``. Used to parse web URLs into ``hf://`` URIs and to decide which
# redirects to follow when resolving files. The auth header is forwarded to these hosts: only add trusted ones.
HF_URL_HOSTS: frozenset[str] = frozenset(
    {"hf.co"}
    | {
        host.lower()
        for host in (
            urlsplit(_HF_DEFAULT_ENDPOINT).hostname,
            urlsplit(_HF_DEFAULT_STAGING_ENDPOINT).hostname,
            urlsplit(ENDPOINT).hostname,
        )
        if host
    }
)

DATASETS_SERVER_ENDPOINT = "https://datasets-server.huggingface.co"

HUGGINGFACE_HEADER_X_REPO_COMMIT = "X-Repo-Commit"
HUGGINGFACE_HEADER_X_LINKED_ETAG = "X-Linked-Etag"
HUGGINGFACE_HEADER_X_LINKED_SIZE = "X-Linked-Size"
HUGGINGFACE_HEADER_X_BILL_TO = "X-HF-Bill-To"
# Header sent on every HTTP call made while downloading files from the Hub (see `flag_as_download_call`).
X_HF_DOWNLOAD_COUNTER = "X-HF-Download-Counter"

INFERENCE_ENDPOINT = os.environ.get("HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co")

# See https://huggingface.co/docs/inference-endpoints/index
INFERENCE_ENDPOINTS_ENDPOINT = "https://api.endpoints.huggingface.cloud/v2"
INFERENCE_CATALOG_ENDPOINT = "https://endpoints.huggingface.co/api/v1/catalog"

# Proxy for third-party providers
INFERENCE_PROXY_TEMPLATE = "https://router.huggingface.co/{provider}"

REPO_ID_SEPARATOR = "--"
# ^ this substring is not allowed in repo_ids on hf.co
# and is the canonical one we use for serialization of repo ids elsewhere.


REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPE_KERNEL = "kernel"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
REPO_TYPES_WITH_KERNEL = REPO_TYPES + [REPO_TYPE_KERNEL]
SPACES_SDK_TYPES = ["gradio", "streamlit", "docker", "static"]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
    REPO_TYPE_KERNEL: "kernels/",
}
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
    "kernels": REPO_TYPE_KERNEL,
}

# HF Hub URIs (``hf://...``). See ``huggingface_hub/utils/_hf_uris.py``
# and ``docs/source/en/package_reference/hf_uris.md`` for the full grammar.
HF_PROTOCOL = "hf://"
HfUriType = Literal["model", "dataset", "space", "kernel", "bucket"]
# Maps the plural URI prefix that may appear in a HF URI (e.g. ``datasets/``)
# to the canonical singular type name. Buckets are first-class HF URI types.
HF_URI_TYPE_PREFIXES: dict[str, HfUriType] = {
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
    "kernels": "kernel",
    "buckets": "bucket",
}


DiscussionTypeFilter = Literal["all", "discussion", "pull_request"]
DISCUSSION_TYPES: tuple[DiscussionTypeFilter, ...] = typing.get_args(DiscussionTypeFilter)
DiscussionStatusFilter = Literal["all", "open", "closed"]
DISCUSSION_STATUS: tuple[DiscussionTypeFilter, ...] = typing.get_args(DiscussionStatusFilter)

# Webhook subscription types
WEBHOOK_DOMAIN_T = Literal["repo", "discussions"]

# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
HF_HOME = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
        )
    )
)

default_cache_path = os.path.join(HF_HOME, "hub")
default_assets_cache_path = os.path.join(HF_HOME, "assets")

# Legacy env variables
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
HUGGINGFACE_ASSETS_CACHE = os.getenv("HUGGINGFACE_ASSETS_CACHE", default_assets_cache_path)

# New env variables
HF_HUB_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HUB_CACHE",
            HUGGINGFACE_HUB_CACHE,
        )
    )
)
HF_ASSETS_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_ASSETS_CACHE",
            HUGGINGFACE_ASSETS_CACHE,
        )
    )
)

HF_HUB_OFFLINE = _is_true(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))


def is_offline_mode() -> bool:
    """Returns whether we are in offline mode for the Hub.

    When offline mode is enabled, all HTTP requests made with `get_session` will raise an `OfflineModeIsEnabled` exception.

    Example:
        ```py
        from huggingface_hub import is_offline_mode

        def list_files(repo_id: str):
            if is_offline_mode():
                ... # list files from local cache (degraded experience but still functional)
            else:
                ... # list files from Hub (complete experience)
        ```
    """
    return HF_HUB_OFFLINE


# File created to mark that the version check has been done.
# Check is performed once per 24 hours at most.
CHECK_FOR_UPDATE_DONE_PATH = os.path.join(HF_HOME, ".check_for_update_done")

# File created to mark that the `hf-cli` skill check has been done.
# Check is performed once per 24 hours at most.
CHECK_FOR_SKILL_UPDATE_DONE_PATH = os.path.join(HF_HOME, ".check_for_skill_update_done")

# File caching the AI agent harnesses registry fetched from `{ENDPOINT}/api/agent-harnesses`.
# Refreshed once per 24 hours at most (see `utils/_detect_agent.py`).
AGENT_HARNESSES_PATH = os.path.join(HF_HOME, ".agent_harnesses.json")

# Skills directories for AI agents: `.agents/skills` is read by most agents, `.claude/skills` by Claude Code.
# Local = current project, global = user-level (see `hf skills add`).
# Claude Code's user-level directory defaults to `~/.claude` but can be relocated with `CLAUDE_CONFIG_DIR`.
AGENTS_SKILLS_LOCAL_PATH = Path(".agents/skills")
AGENTS_SKILLS_GLOBAL_PATH = Path("~/.agents/skills")
CLAUDE_SKILLS_LOCAL_PATH = Path(".claude/skills")
CLAUDE_SKILLS_GLOBAL_PATH = Path(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude") / "skills"

# Set to skip the CLI update check (PyPI query + "new version available" warning at startup).
HF_HUB_DISABLE_UPDATE_CHECK = _is_true(os.environ.get("HF_HUB_DISABLE_UPDATE_CHECK"))

# If set, log level will be set to DEBUG and all requests made to the Hub will be logged
# as curl commands for reproducibility.
HF_DEBUG = _is_true(os.environ.get("HF_DEBUG"))

# Opt-out from telemetry requests
HF_HUB_DISABLE_TELEMETRY = (
    _is_true(os.environ.get("HF_HUB_DISABLE_TELEMETRY"))  # HF-specific env variable
    or _is_true(os.environ.get("DISABLE_TELEMETRY"))
    or _is_true(os.environ.get("DO_NOT_TRACK"))  # https://donottrack.sh/
)

HF_TOKEN_PATH = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_TOKEN_PATH",
            os.path.join(HF_HOME, "token"),
        )
    )
)
HF_STORED_TOKENS_PATH = os.path.join(os.path.dirname(HF_TOKEN_PATH), "stored_tokens")

if _staging_mode:
    # In staging mode, we use a different cache to ensure we don't mix up production and staging data or tokens
    # In practice in `huggingface_hub` tests, we monkeypatch these values with temporary directories. The following
    # lines are only used in third-party libraries tests (e.g. `transformers`, `diffusers`, etc.).
    _staging_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface_staging")
    HUGGINGFACE_HUB_CACHE = os.path.join(_staging_home, "hub")
    HF_TOKEN_PATH = os.path.join(_staging_home, "token")

# Here, `True` will disable progress bars globally without possibility of enabling it
# programmatically. `False` will enable them without possibility of disabling them.
# If environment variable is not set (None), then the user is free to enable/disable
# them programmatically.
# TL;DR: env variable has priority over code
__HF_HUB_DISABLE_PROGRESS_BARS = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
HF_HUB_DISABLE_PROGRESS_BARS: bool | None = (
    _is_true(__HF_HUB_DISABLE_PROGRESS_BARS) if __HF_HUB_DISABLE_PROGRESS_BARS is not None else None
)

# Disable symlinks in the cache (files are copied instead of symlinked)
HF_HUB_DISABLE_SYMLINKS: bool = _is_true(os.environ.get("HF_HUB_DISABLE_SYMLINKS"))

# Disable warning on machines that do not support symlinks (e.g. Windows non-developer)
HF_HUB_DISABLE_SYMLINKS_WARNING: bool = _is_true(os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING"))

# Disable warning when using experimental features
HF_HUB_DISABLE_EXPERIMENTAL_WARNING: bool = _is_true(os.environ.get("HF_HUB_DISABLE_EXPERIMENTAL_WARNING"))

# Disable sending the cached token by default is all HTTP requests to the Hub
HF_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"))

HF_XET_HIGH_PERFORMANCE: bool = _is_true(os.environ.get("HF_XET_HIGH_PERFORMANCE"))

# Bucket and mount path used when launching Jobs
HF_JOBS_ARTIFACTS_BUCKET_NAME: str = "jobs-artifacts"
HF_JOBS_ARTIFACTS_MOUNT_PATH: str = "/data"

# hf_transfer is not used anymore. Let's warn user is case they set the env variable.
# Note: we use FutureWarning (shown by default) instead of DeprecationWarning (silenced
# by default for end users) so users running standard `python` actually see the message.
if _is_true(os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")) and not HF_XET_HIGH_PERFORMANCE:
    import warnings

    warnings.warn(
        "The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. "
        "Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. "
        "Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.",
        FutureWarning,
    )

# Used to override the etag timeout on a system level
HF_HUB_ETAG_TIMEOUT: int = _as_int(os.environ.get("HF_HUB_ETAG_TIMEOUT")) or DEFAULT_ETAG_TIMEOUT

# Used to override the get request timeout on a system level
# Also used as a default timeout for other requests if not specified (kept the naming for legacy reasons)
HF_HUB_DOWNLOAD_TIMEOUT: int = _as_int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")) or DEFAULT_DOWNLOAD_TIMEOUT

# Allows to add information about the requester in the user-agent (e.g. partner name)
HF_HUB_USER_AGENT_ORIGIN: str | None = os.environ.get("HF_HUB_USER_AGENT_ORIGIN")

# If OAuth didn't work after 2 redirects, there's likely a third-party cookie issue in the Space iframe view.
# In this case, we redirect the user to the non-iframe view.
OAUTH_MAX_REDIRECTS = 2

# OAuth-related environment variables injected by the Space
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL")

# OAuth client ID of the Device Code login flow (RFC 8628) used by `hf auth login` / `login()`.
# Overridable for Hub deployments (staging, Enterprise) where the default client ID is not provisioned.
DEVICE_CODE_OAUTH_CLIENT_ID = os.environ.get("HF_DEVICE_CODE_OAUTH_CLIENT_ID", "26be6b09-91c5-47da-9861-d2d2bb7a7e36")

# Xet constants
HUGGINGFACE_HEADER_X_XET_ENDPOINT = "X-Xet-Cas-Url"
HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN = "X-Xet-Access-Token"
HUGGINGFACE_HEADER_X_XET_EXPIRATION = "X-Xet-Token-Expiration"
HUGGINGFACE_HEADER_X_XET_HASH = "X-Xet-Hash"
HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE = "X-Xet-Refresh-Route"
HUGGINGFACE_HEADER_LINK_XET_AUTH_KEY = "xet-auth"

default_xet_cache_path = os.path.join(HF_HOME, "xet")
HF_XET_CACHE = os.getenv("HF_XET_CACHE", default_xet_cache_path)
HF_HUB_DISABLE_XET: bool = _is_true(os.environ.get("HF_HUB_DISABLE_XET"))

# Disable the cache-wide shared blob store (see `huggingface_hub.utils._shared_blobs`).
HF_HUB_DISABLE_SHARED_BLOBS: bool = _is_true(os.environ.get("HF_HUB_DISABLE_SHARED_BLOBS"))

# Bucket hosting the static sandbox server binary (see huggingface_hub.Sandbox)
SANDBOX_SERVER_BUCKET: str = "huggingface/sbx-server"
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
"""On-disk cache for repository tree listings.

A tree listing is the set of files (with their download metadata) contained in a repo at a given commit. Because a
commit hash is immutable, its tree listing never changes and can be cached forever without any invalidation logic.

The listing is stored as a human-readable JSON file under `<tree_cache_folder>/trees/<commit_hash>.json`. The
folder depends on the download target: the per-repo `storage_folder` for `cache_dir` downloads, or
`local_dir/.cache/huggingface/` for `local_dir` downloads (see `tree_cache_folder_for_local_dir`).

```json
{
  "format_version": 1,
  "files": {
    "config.json": {"size": 519, "blob_id": "<git sha1>"},
    "model.safetensors": {"size": 1234, "blob_id": "...", "lfs_sha256": "<sha256>", "lfs_size": 1234, "xet_hash": "..."}
  }
}
```
"""

import json
import os
import tempfile
import threading
from dataclasses import dataclass

from .utils import logging
from .utils._paths import as_extended_path
from .utils._xet import is_valid_xet_hash


logger = logging.get_logger(__name__)

TREE_CACHE_FORMAT_VERSION = 1

# In-memory cache of parsed tree listings, keyed by absolute file path.
_IN_MEMORY_TREE_CACHE: dict[str, "dict[str, TreeCacheEntry]"] = {}
_IN_MEMORY_TREE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class TreeCacheEntry:
    """Raw metadata of a single file in a cached tree listing, mirroring the `/tree` endpoint fields."""

    size: int
    blob_id: str
    lfs_sha256: str | None = None
    lfs_size: int | None = None
    xet_hash: str | None = None

    def to_json(self) -> dict:
        info: dict = {"size": self.size, "blob_id": self.blob_id}
        if self.lfs_sha256 is not None:
            info["lfs_sha256"] = self.lfs_sha256
            info["lfs_size"] = self.lfs_size
        if self.xet_hash is not None:
            info["xet_hash"] = self.xet_hash
        return info

    @classmethod
    def from_json(cls, info: dict) -> "TreeCacheEntry":
        return cls(
            size=info["size"],
            blob_id=info["blob_id"],
            lfs_sha256=info.get("lfs_sha256"),
            lfs_size=info.get("lfs_size"),
            xet_hash=info.get("xet_hash"),
        )


def is_valid_tree_entries(entries: dict[str, TreeCacheEntry]) -> bool:
    """Return whether all Xet hashes in the tree listing are valid."""
    return all(entry.xet_hash is None or is_valid_xet_hash(entry.xet_hash) for entry in entries.values())


def _tree_cache_path(tree_cache_folder: str, commit_hash: str) -> str:
    # Both readers and writers go through here, so they agree on the path (and on the in-memory cache key), including
    # on Windows where a deep `local_dir`/`cache_dir` requires the extended-length form.
    return as_extended_path(os.path.join(tree_cache_folder, "trees", f"{commit_hash}.json"))


def tree_cache_folder_for_local_dir(local_dir: str) -> str:
    """Folder under which the `trees/` cache lives for a `local_dir` download."""
    return os.path.join(local_dir, ".cache", "huggingface")


def read_tree_cache(tree_cache_folder: str, commit_hash: str) -> dict[str, TreeCacheEntry] | None:
    """Return the cached tree listing for a commit hash, or `None` if not cached, invalid, or unreadable."""
    path = _tree_cache_path(tree_cache_folder, commit_hash)
    with _IN_MEMORY_TREE_CACHE_LOCK:
        if path in _IN_MEMORY_TREE_CACHE:
            cached_entries = _IN_MEMORY_TREE_CACHE[path]
            return cached_entries if is_valid_tree_entries(cached_entries) else None
    entries = _read_tree_cache_from_disk(path)
    if entries is None or not is_valid_tree_entries(entries):
        return None
    with _IN_MEMORY_TREE_CACHE_LOCK:
        _IN_MEMORY_TREE_CACHE[path] = entries
    return entries


def _read_tree_cache_from_disk(path: str) -> dict[str, TreeCacheEntry] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format_version") != TREE_CACHE_FORMAT_VERSION:
            # Unknown format (e.g. written by a newer version) => ignore and re-fetch.
            return None
        return {file_path: TreeCacheEntry.from_json(info) for file_path, info in data["files"].items()}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Ignoring corrupted tree cache file {path}: {e}")
        return None


def write_tree_cache(tree_cache_folder: str, commit_hash: str, entries: dict[str, TreeCacheEntry]) -> None:
    """Write a valid tree listing to the cache (ignoring invalid entries and any failures)."""
    if not is_valid_tree_entries(entries):
        return

    path = _tree_cache_path(tree_cache_folder, commit_hash)
    data = {
        "format_version": TREE_CACHE_FORMAT_VERSION,
        "files": {file_path: entries[file_path].to_json() for file_path in sorted(entries)},
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(f"Ignored error while writing tree cache file {path}: {e}")
        return

    # Seed the in-memory cache so later readers of this commit skip re-reading and re-parsing the file.
    with _IN_MEMORY_TREE_CACHE_LOCK:
        _IN_MEMORY_TREE_CACHE[path] = dict(entries)
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

# ***********
# `huggingface_hub` init has 2 modes:
# - Normal usage:
#       If imported to use it, all modules and functions are lazy-loaded. This means
#       they exist at top level in module but are imported only the first time they are
#       used. This way, `from huggingface_hub import something` will import `something`
#       quickly without the hassle of importing all the features from `huggingface_hub`.
# - Static check:
#       If statically analyzed, all modules and functions are loaded normally. This way
#       static typing check works properly as well as autocomplete in text editors and
#       IDEs.
#
# The static model imports are done inside the `if TYPE_CHECKING:` statement at
# the bottom of this file. Since module/functions imports are duplicated, it is
# mandatory to make sure to add them twice when adding one. This is checked in the
# `make quality` command.
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_static_imports.py --update-file
#
# # Or run style on codebase
# make style
# ```
#
# ***********
# Lazy loader vendored from https://github.com/scientific-python/lazy_loader
import importlib
import os
import sys
from typing import TYPE_CHECKING


__version__ = "2.2.0"

# Alphabetical order of definitions is ensured in tests
# WARNING: any comment added in this dictionary definition will be lost when
# re-generating the file !
_SUBMOD_ATTRS = {
    "_buckets": [
        "BucketFile",
        "BucketFileMetadata",
        "BucketFolder",
        "BucketInfo",
        "BucketUrl",
        "SyncOperation",
        "SyncPlan",
    ],
    "_commit_api": [
        "CommitOperation",
        "CommitOperationAdd",
        "CommitOperationCopy",
        "CommitOperationDelete",
    ],
    "_commit_scheduler": [
        "CommitScheduler",
    ],
    "_eval_results": [
        "EvalResultEntry",
        "eval_result_entries_to_yaml",
        "parse_eval_result_entries",
    ],
    "_inference_endpoints": [
        "InferenceCatalogModel",
        "InferenceCatalogRecipe",
        "InferenceEndpoint",
        "InferenceEndpointError",
        "InferenceEndpointHardware",
        "InferenceEndpointStatus",
        "InferenceEndpointTimeoutError",
        "InferenceEndpointType",
    ],
    "_jobs_api": [
        "JobAccelerator",
        "JobDurations",
        "JobHardware",
        "JobHardwareInfo",
        "JobInfo",
        "JobInitiator",
        "JobNetwork",
        "JobOwner",
        "JobStage",
        "JobStatus",
    ],
    "_login": [
        "auth_list",
        "auth_switch",
        "interpreter_login",
        "login",
        "logout",
        "notebook_login",
    ],
    "_oauth": [
        "OAuthInfo",
        "OAuthOrgInfo",
        "OAuthUserInfo",
        "attach_huggingface_oauth",
        "parse_huggingface_oauth",
    ],
    "_revision": [
        "ResolvedRevision",
    ],
    "_sandbox": [
        "Sandbox",
        "SandboxCommandResult",
        "SandboxPool",
        "SandboxProcess",
    ],
    "_snapshot_download": [
        "get_cached_repo_tree",
        "snapshot_download",
    ],
    "_space_api": [
        "SpaceHardware",
        "SpaceRuntime",
        "SpaceSecret",
        "SpaceStage",
        "SpaceStorage",
        "SpaceVariable",
        "Volume",
        "ZeroGpuQuota",
    ],
    "_tensorboard_logger": [
        "HFSummaryWriter",
    ],
    "_webhooks_payload": [
        "WebhookPayload",
        "WebhookPayloadComment",
        "WebhookPayloadDiscussion",
        "WebhookPayloadDiscussionChanges",
        "WebhookPayloadEvent",
        "WebhookPayloadMovedTo",
        "WebhookPayloadRepo",
        "WebhookPayloadUrl",
        "WebhookPayloadWebhook",
    ],
    "_webhooks_server": [
        "WebhooksServer",
        "webhook_endpoint",
    ],
    "cli._cli_utils": [
        "check_cli_update",
        "typer_factory",
    ],
    "community": [
        "Discussion",
        "DiscussionComment",
        "DiscussionCommit",
        "DiscussionEvent",
        "DiscussionStatusChange",
        "DiscussionTitleChange",
        "DiscussionWithDetails",
    ],
    "constants": [
        "CONFIG_NAME",
        "FLAX_WEIGHTS_NAME",
        "HUGGINGFACE_CO_URL_HOME",
        "HUGGINGFACE_CO_URL_TEMPLATE",
        "PYTORCH_WEIGHTS_NAME",
        "REPO_TYPE_DATASET",
        "REPO_TYPE_MODEL",
        "REPO_TYPE_SPACE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "is_offline_mode",
    ],
    "fastai_utils": [
        "_save_pretrained_fastai",
        "from_pretrained_fastai",
        "push_to_hub_fastai",
    ],
    "file_download": [
        "DryRunFileInfo",
        "HfFileMetadata",
        "_CACHED_NO_EXIST",
        "get_hf_file_metadata",
        "hf_hub_download",
        "hf_hub_url",
        "try_to_load_from_cache",
    ],
    "hf_api": [
        "Collection",
        "CollectionItem",
        "CommitInfo",
        "DatasetInfo",
        "DatasetLeaderboardEntry",
        "DuplicatedRepoUrl",
        "GitCommitInfo",
        "GitRefInfo",
        "GitRefs",
        "HfApi",
        "KernelInfo",
        "ModelInfo",
        "Organization",
        "RepoFile",
        "RepoFolder",
        "RepoStorageInfo",
        "RepoUrl",
        "SpaceInfo",
        "SpaceSearchResult",
        "SpaceTemplate",
        "User",
        "UserLikes",
        "WebhookInfo",
        "WebhookWatchedItem",
        "accept_access_request",
        "add_collection_item",
        "add_space_secret",
        "add_space_variable",
        "auth_check",
        "batch_bucket_files",
        "bucket_info",
        "cancel_access_request",
        "cancel_job",
        "change_discussion_status",
        "comment_discussion",
        "copy_files",
        "create_branch",
        "create_bucket",
        "create_collection",
        "create_commit",
        "create_discussion",
        "create_inference_endpoint",
        "create_inference_endpoint_from_catalog",
        "create_pull_request",
        "create_repo",
        "create_scheduled_job",
        "create_scheduled_uv_job",
        "create_tag",
        "create_webhook",
        "dataset_info",
        "delete_branch",
        "delete_bucket",
        "delete_collection",
        "delete_collection_item",
        "delete_file",
        "delete_folder",
        "delete_inference_endpoint",
        "delete_repo",
        "delete_scheduled_job",
        "delete_space_secret",
        "delete_space_variable",
        "delete_space_volumes",
        "delete_tag",
        "delete_webhook",
        "disable_space_dev_mode",
        "disable_webhook",
        "download_bucket_files",
        "duplicate_repo",
        "edit_discussion_comment",
        "enable_space_dev_mode",
        "enable_webhook",
        "fetch_job_logs",
        "fetch_job_metrics",
        "fetch_space_logs",
        "file_exists",
        "get_bucket_file_metadata",
        "get_bucket_paths_info",
        "get_collection",
        "get_dataset_leaderboard",
        "get_dataset_tags",
        "get_discussion_details",
        "get_full_repo_name",
        "get_inference_endpoint",
        "get_local_safetensors_metadata",
        "get_model_tags",
        "get_organization_overview",
        "get_paths_info",
        "get_repo_discussions",
        "get_safetensors_metadata",
        "get_space_runtime",
        "get_space_secrets",
        "get_space_variables",
        "get_user_overview",
        "get_webhook",
        "get_zero_gpu_quota",
        "grant_access",
        "inspect_job",
        "inspect_scheduled_job",
        "kernel_info",
        "list_accepted_access_requests",
        "list_bucket_tree",
        "list_buckets",
        "list_collections",
        "list_daily_papers",
        "list_dataset_parquet_files",
        "list_datasets",
        "list_inference_catalog",
        "list_inference_endpoints",
        "list_inference_endpoints_hardware",
        "list_jobs",
        "list_jobs_hardware",
        "list_lfs_files",
        "list_liked_repos",
        "list_models",
        "list_organization_followers",
        "list_organization_members",
        "list_papers",
        "list_pending_access_requests",
        "list_rejected_access_requests",
        "list_repo_commits",
        "list_repo_files",
        "list_repo_likers",
        "list_repo_refs",
        "list_repo_tree",
        "list_space_templates",
        "list_spaces",
        "list_spaces_hardware",
        "list_user_followers",
        "list_user_following",
        "list_user_repos",
        "list_webhooks",
        "merge_pull_request",
        "model_info",
        "move_bucket",
        "move_repo",
        "paper_info",
        "parse_local_safetensors_file_metadata",
        "parse_safetensors_file_metadata",
        "pause_inference_endpoint",
        "pause_space",
        "permanently_delete_lfs_files",
        "preupload_lfs_files",
        "read_paper",
        "reject_access_request",
        "rename_discussion",
        "repo_exists",
        "repo_info",
        "request_space_hardware",
        "rerun_job",
        "resolve_revision",
        "restart_space",
        "resume_inference_endpoint",
        "resume_scheduled_job",
        "revision_exists",
        "run_as_future",
        "run_job",
        "run_uv_job",
        "scale_to_zero_inference_endpoint",
        "search_spaces",
        "set_space_sleep_time",
        "set_space_volumes",
        "space_info",
        "super_squash_history",
        "suspend_scheduled_job",
        "sync_bucket",
        "sync_job_volume",
        "trigger_scheduled_job",
        "unlike",
        "update_bucket_settings",
        "update_collection_item",
        "update_collection_metadata",
        "update_collection_resource_group",
        "update_inference_endpoint",
        "update_job_expose",
        "update_job_labels",
        "update_repo_settings",
        "update_scheduled_job_labels",
        "update_scheduled_job_schedule",
        "update_webhook",
        "upload_file",
        "upload_folder",
        "verify_repo_checksums",
        "wait_for_job",
        "wait_for_space",
        "whoami",
    ],
    "hf_file_system": [
        "HfFileSystem",
        "HfFileSystemFile",
        "HfFileSystemResolvedPath",
        "HfFileSystemStreamFile",
        "hffs",
    ],
    "hub_mixin": [
        "ModelHubMixin",
        "PyTorchModelHubMixin",
    ],
    "inference._client": [
        "InferenceClient",
        "InferenceTimeoutError",
    ],
    "inference._generated._async_client": [
        "AsyncInferenceClient",
    ],
    "inference._generated.types": [
        "AudioClassificationInput",
        "AudioClassificationOutputElement",
        "AudioClassificationOutputTransform",
        "AudioClassificationParameters",
        "AudioToAudioInput",
        "AudioToAudioOutputElement",
        "AutomaticSpeechRecognitionEarlyStoppingEnum",
        "AutomaticSpeechRecognitionGenerationParameters",
        "AutomaticSpeechRecognitionInput",
        "AutomaticSpeechRecognitionOutput",
        "AutomaticSpeechRecognitionOutputChunk",
        "AutomaticSpeechRecognitionParameters",
        "ChatCompletionInput",
        "ChatCompletionInputFunctionDefinition",
        "ChatCompletionInputFunctionName",
        "ChatCompletionInputGrammarType",
        "ChatCompletionInputJSONSchema",
        "ChatCompletionInputMessage",
        "ChatCompletionInputMessageChunk",
        "ChatCompletionInputMessageChunkType",
        "ChatCompletionInputResponseFormatJSONObject",
        "ChatCompletionInputResponseFormatJSONSchema",
        "ChatCompletionInputResponseFormatText",
        "ChatCompletionInputStreamOptions",
        "ChatCompletionInputTool",
        "ChatCompletionInputToolCall",
        "ChatCompletionInputToolChoiceClass",
        "ChatCompletionInputToolChoiceEnum",
        "ChatCompletionInputURL",
        "ChatCompletionOutput",
        "ChatCompletionOutputComplete",
        "ChatCompletionOutputFunctionDefinition",
        "ChatCompletionOutputLogprob",
        "ChatCompletionOutputLogprobs",
        "ChatCompletionOutputMessage",
        "ChatCompletionOutputToolCall",
        "ChatCompletionOutputTopLogprob",
        "ChatCompletionOutputUsage",
        "ChatCompletionStreamOutput",
        "ChatCompletionStreamOutputChoice",
        "ChatCompletionStreamOutputDelta",
        "ChatCompletionStreamOutputDeltaToolCall",
        "ChatCompletionStreamOutputFunction",
        "ChatCompletionStreamOutputLogprob",
        "ChatCompletionStreamOutputLogprobs",
        "ChatCompletionStreamOutputTopLogprob",
        "ChatCompletionStreamOutputUsage",
        "DepthEstimationInput",
        "DepthEstimationOutput",
        "DocumentQuestionAnsweringInput",
        "DocumentQuestionAnsweringInputData",
        "DocumentQuestionAnsweringOutputElement",
        "DocumentQuestionAnsweringParameters",
        "FeatureExtractionInput",
        "FeatureExtractionInputTruncationDirection",
        "FillMaskInput",
        "FillMaskOutputElement",
        "FillMaskParameters",
        "ImageClassificationInput",
        "ImageClassificationOutputElement",
        "ImageClassificationOutputTransform",
        "ImageClassificationParameters",
        "ImageSegmentationInput",
        "ImageSegmentationOutputElement",
        "ImageSegmentationParameters",
        "ImageSegmentationSubtask",
        "ImageTextToImageInput",
        "ImageTextToImageOutput",
        "ImageTextToImageParameters",
        "ImageTextToImageTargetSize",
        "ImageTextToVideoInput",
        "ImageTextToVideoOutput",
        "ImageTextToVideoParameters",
        "ImageTextToVideoTargetSize",
        "ImageToImageInput",
        "ImageToImageOutput",
        "ImageToImageParameters",
        "ImageToImageTargetSize",
        "ImageToTextEarlyStoppingEnum",
        "ImageToTextGenerationParameters",
        "ImageToTextInput",
        "ImageToTextOutput",
        "ImageToTextParameters",
        "ImageToVideoInput",
        "ImageToVideoOutput",
        "ImageToVideoParameters",
        "ImageToVideoTargetSize",
        "ObjectDetectionBoundingBox",
        "ObjectDetectionInput",
        "ObjectDetectionOutputElement",
        "ObjectDetectionParameters",
        "Padding",
        "QuestionAnsweringInput",
        "QuestionAnsweringInputData",
        "QuestionAnsweringOutputElement",
        "QuestionAnsweringParameters",
        "SentenceSimilarityInput",
        "SentenceSimilarityInputData",
        "SummarizationInput",
        "SummarizationOutput",
        "SummarizationParameters",
        "SummarizationTruncationStrategy",
        "TableQuestionAnsweringInput",
        "TableQuestionAnsweringInputData",
        "TableQuestionAnsweringOutputElement",
        "TableQuestionAnsweringParameters",
        "Text2TextGenerationInput",
        "Text2TextGenerationOutput",
        "Text2TextGenerationParameters",
        "Text2TextGenerationTruncationStrategy",
        "TextClassificationInput",
        "TextClassificationOutputElement",
        "TextClassificationOutputTransform",
        "TextClassificationParameters",
        "TextGenerationInput",
        "TextGenerationInputGenerateParameters",
        "TextGenerationInputGrammarType",
        "TextGenerationOutput",
        "TextGenerationOutputBestOfSequence",
        "TextGenerationOutputDetails",
        "TextGenerationOutputFinishReason",
        "TextGenerationOutputPrefillToken",
        "TextGenerationOutputToken",
        "TextGenerationStreamOutput",
        "TextGenerationStreamOutputStreamDetails",
        "TextGenerationStreamOutputToken",
        "TextToAudioEarlyStoppingEnum",
        "TextToAudioGenerationParameters",
        "TextToAudioInput",
        "TextToAudioOutput",
        "TextToAudioParameters",
        "TextToImageInput",
        "TextToImageOutput",
        "TextToImageParameters",
        "TextToSpeechEarlyStoppingEnum",
        "TextToSpeechGenerationParameters",
        "TextToSpeechInput",
        "TextToSpeechOutput",
        "TextToSpeechParameters",
        "TextToVideoInput",
        "TextToVideoOutput",
        "TextToVideoParameters",
        "TokenClassificationAggregationStrategy",
        "TokenClassificationInput",
        "TokenClassificationOutputElement",
        "TokenClassificationParameters",
        "TranslationInput",
        "TranslationOutput",
        "TranslationParameters",
        "TranslationTruncationStrategy",
        "TypeEnum",
        "VideoClassificationInput",
        "VideoClassificationOutputElement",
        "VideoClassificationOutputTransform",
        "VideoClassificationParameters",
        "VisualQuestionAnsweringInput",
        "VisualQuestionAnsweringInputData",
        "VisualQuestionAnsweringOutputElement",
        "VisualQuestionAnsweringParameters",
        "ZeroShotClassificationInput",
        "ZeroShotClassificationOutputElement",
        "ZeroShotClassificationParameters",
        "ZeroShotImageClassificationInput",
        "ZeroShotImageClassificationOutputElement",
        "ZeroShotImageClassificationParameters",
        "ZeroShotObjectDetectionBoundingBox",
        "ZeroShotObjectDetectionInput",
        "ZeroShotObjectDetectionOutputElement",
        "ZeroShotObjectDetectionParameters",
    ],
    "inference._mcp.agent": [
        "Agent",
    ],
    "inference._mcp.mcp_client": [
        "MCPClient",
    ],
    "repocard": [
        "DatasetCard",
        "ModelCard",
        "RepoCard",
        "SpaceCard",
        "metadata_eval_result",
        "metadata_load",
        "metadata_save",
        "metadata_update",
    ],
    "repocard_data": [
        "CardData",
        "DatasetCardData",
        "EvalResult",
        "ModelCardData",
        "SpaceCardData",
    ],
    "serialization": [
        "StateDictSplit",
        "get_torch_storage_id",
        "get_torch_storage_size",
        "load_state_dict_from_file",
        "load_torch_model",
        "save_torch_model",
        "save_torch_state_dict",
        "split_state_dict_into_shards_factory",
        "split_torch_state_dict_into_shards",
    ],
    "serialization._dduf": [
        "DDUFEntry",
        "export_entries_as_dduf",
        "export_folder_as_dduf",
        "read_dduf_file",
    ],
    "utils": [
        "ASYNC_CLIENT_FACTORY_T",
        "CLIENT_FACTORY_T",
        "CacheNotFound",
        "CachedFileInfo",
        "CachedIncompleteFileInfo",
        "CachedRepoInfo",
        "CachedRevisionInfo",
        "CorruptedCacheException",
        "DeleteCacheStrategy",
        "HFCacheInfo",
        "HfUri",
        "cached_assets_path",
        "close_session",
        "dump_environment_info",
        "get_async_session",
        "get_session",
        "get_token",
        "hf_raise_for_status",
        "logging",
        "parse_hf_uri",
        "scan_cache_dir",
        "set_async_client_factory",
        "set_client_factory",
    ],
}

# WARNING: __all__ is generated automatically, Any manual edit will be lost when re-generating this file !
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_all_variable.py --update
#
# # Or run style on codebase
# make style
# ```

__all__ = [
    "ASYNC_CLIENT_FACTORY_T",
    "Agent",
    "AsyncInferenceClient",
    "AudioClassificationInput",
    "AudioClassificationOutputElement",
    "AudioClassificationOutputTransform",
    "AudioClassificationParameters",
    "AudioToAudioInput",
    "AudioToAudioOutputElement",
    "AutomaticSpeechRecognitionEarlyStoppingEnum",
    "AutomaticSpeechRecognitionGenerationParameters",
    "AutomaticSpeechRecognitionInput",
    "AutomaticSpeechRecognitionOutput",
    "AutomaticSpeechRecognitionOutputChunk",
    "AutomaticSpeechRecognitionParameters",
    "BucketFile",
    "BucketFileMetadata",
    "BucketFolder",
    "BucketInfo",
    "BucketUrl",
    "CLIENT_FACTORY_T",
    "CONFIG_NAME",
    "CacheNotFound",
    "CachedFileInfo",
    "CachedIncompleteFileInfo",
    "CachedRepoInfo",
    "CachedRevisionInfo",
    "CardData",
    "ChatCompletionInput",
    "ChatCompletionInputFunctionDefinition",
    "ChatCompletionInputFunctionName",
    "ChatCompletionInputGrammarType",
    "ChatCompletionInputJSONSchema",
    "ChatCompletionInputMessage",
    "ChatCompletionInputMessageChunk",
    "ChatCompletionInputMessageChunkType",
    "ChatCompletionInputResponseFormatJSONObject",
    "ChatCompletionInputResponseFormatJSONSchema",
    "ChatCompletionInputResponseFormatText",
    "ChatCompletionInputStreamOptions",
    "ChatCompletionInputTool",
    "ChatCompletionInputToolCall",
    "ChatCompletionInputToolChoiceClass",
    "ChatCompletionInputToolChoiceEnum",
    "ChatCompletionInputURL",
    "ChatCompletionOutput",
    "ChatCompletionOutputComplete",
    "ChatCompletionOutputFunctionDefinition",
    "ChatCompletionOutputLogprob",
    "ChatCompletionOutputLogprobs",
    "ChatCompletionOutputMessage",
    "ChatCompletionOutputToolCall",
    "ChatCompletionOutputTopLogprob",
    "ChatCompletionOutputUsage",
    "ChatCompletionStreamOutput",
    "ChatCompletionStreamOutputChoice",
    "ChatCompletionStreamOutputDelta",
    "ChatCompletionStreamOutputDeltaToolCall",
    "ChatCompletionStreamOutputFunction",
    "ChatCompletionStreamOutputLogprob",
    "ChatCompletionStreamOutputLogprobs",
    "ChatCompletionStreamOutputTopLogprob",
    "ChatCompletionStreamOutputUsage",
    "Collection",
    "CollectionItem",
    "CommitInfo",
    "CommitOperation",
    "CommitOperationAdd",
    "CommitOperationCopy",
    "CommitOperationDelete",
    "CommitScheduler",
    "CorruptedCacheException",
    "DDUFEntry",
    "DatasetCard",
    "DatasetCardData",
    "DatasetInfo",
    "DatasetLeaderboardEntry",
    "DeleteCacheStrategy",
    "DepthEstimationInput",
    "DepthEstimationOutput",
    "Discussion",
    "DiscussionComment",
    "DiscussionCommit",
    "DiscussionEvent",
    "DiscussionStatusChange",
    "DiscussionTitleChange",
    "DiscussionWithDetails",
    "DocumentQuestionAnsweringInput",
    "DocumentQuestionAnsweringInputData",
    "DocumentQuestionAnsweringOutputElement",
    "DocumentQuestionAnsweringParameters",
    "DryRunFileInfo",
    "DuplicatedRepoUrl",
    "EvalResult",
    "EvalResultEntry",
    "FLAX_WEIGHTS_NAME",
    "FeatureExtractionInput",
    "FeatureExtractionInputTruncationDirection",
    "FillMaskInput",
    "FillMaskOutputElement",
    "FillMaskParameters",
    "GitCommitInfo",
    "GitRefInfo",
    "GitRefs",
    "HFCacheInfo",
    "HFSummaryWriter",
    "HUGGINGFACE_CO_URL_HOME",
    "HUGGINGFACE_CO_URL_TEMPLATE",
    "HfApi",
    "HfFileMetadata",
    "HfFileSystem",
    "HfFileSystemFile",
    "HfFileSystemResolvedPath",
    "HfFileSystemStreamFile",
    "HfUri",
    "ImageClassificationInput",
    "ImageClassificationOutputElement",
    "ImageClassificationOutputTransform",
    "ImageClassificationParameters",
    "ImageSegmentationInput",
    "ImageSegmentationOutputElement",
    "ImageSegmentationParameters",
    "ImageSegmentationSubtask",
    "ImageTextToImageInput",
    "ImageTextToImageOutput",
    "ImageTextToImageParameters",
    "ImageTextToImageTargetSize",
    "ImageTextToVideoInput",
    "ImageTextToVideoOutput",
    "ImageTextToVideoParameters",
    "ImageTextToVideoTargetSize",
    "ImageToImageInput",
    "ImageToImageOutput",
    "ImageToImageParameters",
    "ImageToImageTargetSize",
    "ImageToTextEarlyStoppingEnum",
    "ImageToTextGenerationParameters",
    "ImageToTextInput",
    "ImageToTextOutput",
    "ImageToTextParameters",
    "ImageToVideoInput",
    "ImageToVideoOutput",
    "ImageToVideoParameters",
    "ImageToVideoTargetSize",
    "InferenceCatalogModel",
    "InferenceCatalogRecipe",
    "InferenceClient",
    "InferenceEndpoint",
    "InferenceEndpointError",
    "InferenceEndpointHardware",
    "InferenceEndpointStatus",
    "InferenceEndpointTimeoutError",
    "InferenceEndpointType",
    "InferenceTimeoutError",
    "JobAccelerator",
    "JobDurations",
    "JobHardware",
    "JobHardwareInfo",
    "JobInfo",
    "JobInitiator",
    "JobNetwork",
    "JobOwner",
    "JobStage",
    "JobStatus",
    "KernelInfo",
    "MCPClient",
    "ModelCard",
    "ModelCardData",
    "ModelHubMixin",
    "ModelInfo",
    "OAuthInfo",
    "OAuthOrgInfo",
    "OAuthUserInfo",
    "ObjectDetectionBoundingBox",
    "ObjectDetectionInput",
    "ObjectDetectionOutputElement",
    "ObjectDetectionParameters",
    "Organization",
    "PYTORCH_WEIGHTS_NAME",
    "Padding",
    "PyTorchModelHubMixin",
    "QuestionAnsweringInput",
    "QuestionAnsweringInputData",
    "QuestionAnsweringOutputElement",
    "QuestionAnsweringParameters",
    "REPO_TYPE_DATASET",
    "REPO_TYPE_MODEL",
    "REPO_TYPE_SPACE",
    "RepoCard",
    "RepoFile",
    "RepoFolder",
    "RepoStorageInfo",
    "RepoUrl",
    "ResolvedRevision",
    "Sandbox",
    "SandboxCommandResult",
    "SandboxPool",
    "SandboxProcess",
    "SentenceSimilarityInput",
    "SentenceSimilarityInputData",
    "SpaceCard",
    "SpaceCardData",
    "SpaceHardware",
    "SpaceInfo",
    "SpaceRuntime",
    "SpaceSearchResult",
    "SpaceSecret",
    "SpaceStage",
    "SpaceStorage",
    "SpaceTemplate",
    "SpaceVariable",
    "StateDictSplit",
    "SummarizationInput",
    "SummarizationOutput",
    "SummarizationParameters",
    "SummarizationTruncationStrategy",
    "SyncOperation",
    "SyncPlan",
    "TF2_WEIGHTS_NAME",
    "TF_WEIGHTS_NAME",
    "TableQuestionAnsweringInput",
    "TableQuestionAnsweringInputData",
    "TableQuestionAnsweringOutputElement",
    "TableQuestionAnsweringParameters",
    "Text2TextGenerationInput",
    "Text2TextGenerationOutput",
    "Text2TextGenerationParameters",
    "Text2TextGenerationTruncationStrategy",
    "TextClassificationInput",
    "TextClassificationOutputElement",
    "TextClassificationOutputTransform",
    "TextClassificationParameters",
    "TextGenerationInput",
    "TextGenerationInputGenerateParameters",
    "TextGenerationInputGrammarType",
    "TextGenerationOutput",
    "TextGenerationOutputBestOfSequence",
    "TextGenerationOutputDetails",
    "TextGenerationOutputFinishReason",
    "TextGenerationOutputPrefillToken",
    "TextGenerationOutputToken",
    "TextGenerationStreamOutput",
    "TextGenerationStreamOutputStreamDetails",
    "TextGenerationStreamOutputToken",
    "TextToAudioEarlyStoppingEnum",
    "TextToAudioGenerationParameters",
    "TextToAudioInput",
    "TextToAudioOutput",
    "TextToAudioParameters",
    "TextToImageInput",
    "TextToImageOutput",
    "TextToImageParameters",
    "TextToSpeechEarlyStoppingEnum",
    "TextToSpeechGenerationParameters",
    "TextToSpeechInput",
    "TextToSpeechOutput",
    "TextToSpeechParameters",
    "TextToVideoInput",
    "TextToVideoOutput",
    "TextToVideoParameters",
    "TokenClassificationAggregationStrategy",
    "TokenClassificationInput",
    "TokenClassificationOutputElement",
    "TokenClassificationParameters",
    "TranslationInput",
    "TranslationOutput",
    "TranslationParameters",
    "TranslationTruncationStrategy",
    "TypeEnum",
    "User",
    "UserLikes",
    "VideoClassificationInput",
    "VideoClassificationOutputElement",
    "VideoClassificationOutputTransform",
    "VideoClassificationParameters",
    "VisualQuestionAnsweringInput",
    "VisualQuestionAnsweringInputData",
    "VisualQuestionAnsweringOutputElement",
    "VisualQuestionAnsweringParameters",
    "Volume",
    "WebhookInfo",
    "WebhookPayload",
    "WebhookPayloadComment",
    "WebhookPayloadDiscussion",
    "WebhookPayloadDiscussionChanges",
    "WebhookPayloadEvent",
    "WebhookPayloadMovedTo",
    "WebhookPayloadRepo",
    "WebhookPayloadUrl",
    "WebhookPayloadWebhook",
    "WebhookWatchedItem",
    "WebhooksServer",
    "ZeroGpuQuota",
    "ZeroShotClassificationInput",
    "ZeroShotClassificationOutputElement",
    "ZeroShotClassificationParameters",
    "ZeroShotImageClassificationInput",
    "ZeroShotImageClassificationOutputElement",
    "ZeroShotImageClassificationParameters",
    "ZeroShotObjectDetectionBoundingBox",
    "ZeroShotObjectDetectionInput",
    "ZeroShotObjectDetectionOutputElement",
    "ZeroShotObjectDetectionParameters",
    "_CACHED_NO_EXIST",
    "_save_pretrained_fastai",
    "accept_access_request",
    "add_collection_item",
    "add_space_secret",
    "add_space_variable",
    "attach_huggingface_oauth",
    "auth_check",
    "auth_list",
    "auth_switch",
    "batch_bucket_files",
    "bucket_info",
    "cached_assets_path",
    "cancel_access_request",
    "cancel_job",
    "change_discussion_status",
    "check_cli_update",
    "close_session",
    "comment_discussion",
    "copy_files",
    "create_branch",
    "create_bucket",
    "create_collection",
    "create_commit",
    "create_discussion",
    "create_inference_endpoint",
    "create_inference_endpoint_from_catalog",
    "create_pull_request",
    "create_repo",
    "create_scheduled_job",
    "create_scheduled_uv_job",
    "create_tag",
    "create_webhook",
    "dataset_info",
    "delete_branch",
    "delete_bucket",
    "delete_collection",
    "delete_collection_item",
    "delete_file",
    "delete_folder",
    "delete_inference_endpoint",
    "delete_repo",
    "delete_scheduled_job",
    "delete_space_secret",
    "delete_space_variable",
    "delete_space_volumes",
    "delete_tag",
    "delete_webhook",
    "disable_space_dev_mode",
    "disable_webhook",
    "download_bucket_files",
    "dump_environment_info",
    "duplicate_repo",
    "edit_discussion_comment",
    "enable_space_dev_mode",
    "enable_webhook",
    "eval_result_entries_to_yaml",
    "export_entries_as_dduf",
    "export_folder_as_dduf",
    "fetch_job_logs",
    "fetch_job_metrics",
    "fetch_space_logs",
    "file_exists",
    "from_pretrained_fastai",
    "get_async_session",
    "get_bucket_file_metadata",
    "get_bucket_paths_info",
    "get_cached_repo_tree",
    "get_collection",
    "get_dataset_leaderboard",
    "get_dataset_tags",
    "get_discussion_details",
    "get_full_repo_name",
    "get_hf_file_metadata",
    "get_inference_endpoint",
    "get_local_safetensors_metadata",
    "get_model_tags",
    "get_organization_overview",
    "get_paths_info",
    "get_repo_discussions",
    "get_safetensors_metadata",
    "get_session",
    "get_space_runtime",
    "get_space_secrets",
    "get_space_variables",
    "get_token",
    "get_torch_storage_id",
    "get_torch_storage_size",
    "get_user_overview",
    "get_webhook",
    "get_zero_gpu_quota",
    "grant_access",
    "hf_hub_download",
    "hf_hub_url",
    "hf_raise_for_status",
    "hffs",
    "inspect_job",
    "inspect_scheduled_job",
    "interpreter_login",
    "is_offline_mode",
    "kernel_info",
    "list_accepted_access_requests",
    "list_bucket_tree",
    "list_buckets",
    "list_collections",
    "list_daily_papers",
    "list_dataset_parquet_files",
    "list_datasets",
    "list_inference_catalog",
    "list_inference_endpoints",
    "list_inference_endpoints_hardware",
    "list_jobs",
    "list_jobs_hardware",
    "list_lfs_files",
    "list_liked_repos",
    "list_models",
    "list_organization_followers",
    "list_organization_members",
    "list_papers",
    "list_pending_access_requests",
    "list_rejected_access_requests",
    "list_repo_commits",
    "list_repo_files",
    "list_repo_likers",
    "list_repo_refs",
    "list_repo_tree",
    "list_space_templates",
    "list_spaces",
    "list_spaces_hardware",
    "list_user_followers",
    "list_user_following",
    "list_user_repos",
    "list_webhooks",
    "load_state_dict_from_file",
    "load_torch_model",
    "logging",
    "login",
    "logout",
    "merge_pull_request",
    "metadata_eval_result",
    "metadata_load",
    "metadata_save",
    "metadata_update",
    "model_info",
    "move_bucket",
    "move_repo",
    "notebook_login",
    "paper_info",
    "parse_eval_result_entries",
    "parse_hf_uri",
    "parse_huggingface_oauth",
    "parse_local_safetensors_file_metadata",
    "parse_safetensors_file_metadata",
    "pause_inference_endpoint",
    "pause_space",
    "permanently_delete_lfs_files",
    "preupload_lfs_files",
    "push_to_hub_fastai",
    "read_dduf_file",
    "read_paper",
    "reject_access_request",
    "rename_discussion",
    "repo_exists",
    "repo_info",
    "request_space_hardware",
    "rerun_job",
    "resolve_revision",
    "restart_space",
    "resume_inference_endpoint",
    "resume_scheduled_job",
    "revision_exists",
    "run_as_future",
    "run_job",
    "run_uv_job",
    "save_torch_model",
    "save_torch_state_dict",
    "scale_to_zero_inference_endpoint",
    "scan_cache_dir",
    "search_spaces",
    "set_async_client_factory",
    "set_client_factory",
    "set_space_sleep_time",
    "set_space_volumes",
    "snapshot_download",
    "space_info",
    "split_state_dict_into_shards_factory",
    "split_torch_state_dict_into_shards",
    "super_squash_history",
    "suspend_scheduled_job",
    "sync_bucket",
    "sync_job_volume",
    "trigger_scheduled_job",
    "try_to_load_from_cache",
    "typer_factory",
    "unlike",
    "update_bucket_settings",
    "update_collection_item",
    "update_collection_metadata",
    "update_collection_resource_group",
    "update_inference_endpoint",
    "update_job_expose",
    "update_job_labels",
    "update_repo_settings",
    "update_scheduled_job_labels",
    "update_scheduled_job_schedule",
    "update_webhook",
    "upload_file",
    "upload_folder",
    "verify_repo_checksums",
    "wait_for_job",
    "wait_for_space",
    "webhook_endpoint",
    "whoami",
]


def _attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows:

    ```py
    import mysubmodule
    import anothersubmodule

    from .foo import someattr
    ```

    The idea is to replace a package's `__getattr__`, `__dir__`, such that all imports
    work exactly the way they would with normal imports, except that the import occurs
    upon first use.

    The typical way to call this function, replacing the above imports, is:

    ```python
    __getattr__, __dir__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': ['someattr']}
    )
    ```
    This functionality requires Python 3.7 or higher.

    Args:
        package_name (`str`):
            Typically use `__name__`.
        submodules (`set`):
            List of submodules to attach.
        submod_attrs (`dict`):
            Dictionary of submodule -> list of attributes / functions.
            These attributes are imported as they are used.

    Returns:
        __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {attr: mod for mod, attrs in submod_attrs.items() for attr in attrs}

    def __getattr__(name):
        if name in submodules:
            try:
                return importlib.import_module(f"{package_name}.{name}")
            except Exception as e:
                print(f"Error importing {package_name}.{name}: {e}")
                raise
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            try:
                submod = importlib.import_module(submod_path)
            except Exception as e:
                print(f"Error importing {submod_path}: {e}")
                raise
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    return __getattr__, __dir__


__getattr__, __dir__ = _attach(__name__, submodules=[], submod_attrs=_SUBMOD_ATTRS)

if os.environ.get("EAGER_IMPORT", ""):
    for attr in __all__:
        __getattr__(attr)

# WARNING: any content below this statement is generated automatically. Any manual edit
# will be lost when re-generating this file !
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_static_imports.py --update
#
# # Or run style on codebase
# make style
# ```
if TYPE_CHECKING:  # pragma: no cover
    from ._buckets import (
        BucketFile,  # noqa: F401
        BucketFileMetadata,  # noqa: F401
        BucketFolder,  # noqa: F401
        BucketInfo,  # noqa: F401
        BucketUrl,  # noqa: F401
        SyncOperation,  # noqa: F401
        SyncPlan,  # noqa: F401
    )
    from ._commit_api import (
        CommitOperation,  # noqa: F401
        CommitOperationAdd,  # noqa: F401
        CommitOperationCopy,  # noqa: F401
        CommitOperationDelete,  # noqa: F401
    )
    from ._commit_scheduler import CommitScheduler  # noqa: F401
    from ._eval_results import (
        EvalResultEntry,  # noqa: F401
        eval_result_entries_to_yaml,  # noqa: F401
        parse_eval_result_entries,  # noqa: F401
    )
    from ._inference_endpoints import (
        InferenceCatalogModel,  # noqa: F401
        InferenceCatalogRecipe,  # noqa: F401
        InferenceEndpoint,  # noqa: F401
        InferenceEndpointError,  # noqa: F401
        InferenceEndpointHardware,  # noqa: F401
        InferenceEndpointStatus,  # noqa: F401
        InferenceEndpointTimeoutError,  # noqa: F401
        InferenceEndpointType,  # noqa: F401
    )
    from ._jobs_api import (
        JobAccelerator,  # noqa: F401
        JobDurations,  # noqa: F401
        JobHardware,  # noqa: F401
        JobHardwareInfo,  # noqa: F401
        JobInfo,  # noqa: F401
        JobInitiator,  # noqa: F401
        JobNetwork,  # noqa: F401
        JobOwner,  # noqa: F401
        JobStage,  # noqa: F401
        JobStatus,  # noqa: F401
    )
    from ._login import (
        auth_list,  # noqa: F401
        auth_switch,  # noqa: F401
        interpreter_login,  # noqa: F401
        login,  # noqa: F401
        logout,  # noqa: F401
        notebook_login,  # noqa: F401
    )
    from ._oauth import (
        OAuthInfo,  # noqa: F401
        OAuthOrgInfo,  # noqa: F401
        OAuthUserInfo,  # noqa: F401
        attach_huggingface_oauth,  # noqa: F401
        parse_huggingface_oauth,  # noqa: F401
    )
    from ._revision import ResolvedRevision  # noqa: F401
    from ._sandbox import (
        Sandbox,  # noqa: F401
        SandboxCommandResult,  # noqa: F401
        SandboxPool,  # noqa: F401
        SandboxProcess,  # noqa: F401
    )
    from ._snapshot_download import (
        get_cached_repo_tree,  # noqa: F401
        snapshot_download,  # noqa: F401
    )
    from ._space_api import (
        SpaceHardware,  # noqa: F401
        SpaceRuntime,  # noqa: F401
        SpaceSecret,  # noqa: F401
        SpaceStage,  # noqa: F401
        SpaceStorage,  # noqa: F401
        SpaceVariable,  # noqa: F401
        Volume,  # noqa: F401
        ZeroGpuQuota,  # noqa: F401
    )
    from ._tensorboard_logger import HFSummaryWriter  # noqa: F401
    from ._webhooks_payload import (
        WebhookPayload,  # noqa: F401
        WebhookPayloadComment,  # noqa: F401
        WebhookPayloadDiscussion,  # noqa: F401
        WebhookPayloadDiscussionChanges,  # noqa: F401
        WebhookPayloadEvent,  # noqa: F401
        WebhookPayloadMovedTo,  # noqa: F401
        WebhookPayloadRepo,  # noqa: F401
        WebhookPayloadUrl,  # noqa: F401
        WebhookPayloadWebhook,  # noqa: F401
    )
    from ._webhooks_server import (
        WebhooksServer,  # noqa: F401
        webhook_endpoint,  # noqa: F401
    )
    from .cli._cli_utils import (
        check_cli_update,  # noqa: F401
        typer_factory,  # noqa: F401
    )
    from .community import (
        Discussion,  # noqa: F401
        DiscussionComment,  # noqa: F401
        DiscussionCommit,  # noqa: F401
        DiscussionEvent,  # noqa: F401
        DiscussionStatusChange,  # noqa: F401
        DiscussionTitleChange,  # noqa: F401
        DiscussionWithDetails,  # noqa: F401
    )
    from .constants import (
        CONFIG_NAME,  # noqa: F401
        FLAX_WEIGHTS_NAME,  # noqa: F401
        HUGGINGFACE_CO_URL_HOME,  # noqa: F401
        HUGGINGFACE_CO_URL_TEMPLATE,  # noqa: F401
        PYTORCH_WEIGHTS_NAME,  # noqa: F401
        REPO_TYPE_DATASET,  # noqa: F401
        REPO_TYPE_MODEL,  # noqa: F401
        REPO_TYPE_SPACE,  # noqa: F401
        TF2_WEIGHTS_NAME,  # noqa: F401
        TF_WEIGHTS_NAME,  # noqa: F401
        is_offline_mode,  # noqa: F401
    )
    from .fastai_utils import (
        _save_pretrained_fastai,  # noqa: F401
        from_pretrained_fastai,  # noqa: F401
        push_to_hub_fastai,  # noqa: F401
    )
    from .file_download import (
        _CACHED_NO_EXIST,  # noqa: F401
        DryRunFileInfo,  # noqa: F401
        HfFileMetadata,  # noqa: F401
        get_hf_file_metadata,  # noqa: F401
        hf_hub_download,  # noqa: F401
        hf_hub_url,  # noqa: F401
        try_to_load_from_cache,  # noqa: F401
    )
    from .hf_api import (
        Collection,  # noqa: F401
        CollectionItem,  # noqa: F401
        CommitInfo,  # noqa: F401
        DatasetInfo,  # noqa: F401
        DatasetLeaderboardEntry,  # noqa: F401
        DuplicatedRepoUrl,  # noqa: F401
        GitCommitInfo,  # noqa: F401
        GitRefInfo,  # noqa: F401
        GitRefs,  # noqa: F401
        HfApi,  # noqa: F401
        KernelInfo,  # noqa: F401
        ModelInfo,  # noqa: F401
        Organization,  # noqa: F401
        RepoFile,  # noqa: F401
        RepoFolder,  # noqa: F401
        RepoStorageInfo,  # noqa: F401
        RepoUrl,  # noqa: F401
        SpaceInfo,  # noqa: F401
        SpaceSearchResult,  # noqa: F401
        SpaceTemplate,  # noqa: F401
        User,  # noqa: F401
        UserLikes,  # noqa: F401
        WebhookInfo,  # noqa: F401
        WebhookWatchedItem,  # noqa: F401
        accept_access_request,  # noqa: F401
        add_collection_item,  # noqa: F401
        add_space_secret,  # noqa: F401
        add_space_variable,  # noqa: F401
        auth_check,  # noqa: F401
        batch_bucket_files,  # noqa: F401
        bucket_info,  # noqa: F401
        cancel_access_request,  # noqa: F401
        cancel_job,  # noqa: F401
        change_discussion_status,  # noqa: F401
        comment_discussion,  # noqa: F401
        copy_files,  # noqa: F401
        create_branch,  # noqa: F401
        create_bucket,  # noqa: F401
        create_collection,  # noqa: F401
        create_commit,  # noqa: F401
        create_discussion,  # noqa: F401
        create_inference_endpoint,  # noqa: F401
        create_inference_endpoint_from_catalog,  # noqa: F401
        create_pull_request,  # noqa: F401
        create_repo,  # noqa: F401
        create_scheduled_job,  # noqa: F401
        create_scheduled_uv_job,  # noqa: F401
        create_tag,  # noqa: F401
        create_webhook,  # noqa: F401
        dataset_info,  # noqa: F401
        delete_branch,  # noqa: F401
        delete_bucket,  # noqa: F401
        delete_collection,  # noqa: F401
        delete_collection_item,  # noqa: F401
        delete_file,  # noqa: F401
        delete_folder,  # noqa: F401
        delete_inference_endpoint,  # noqa: F401
        delete_repo,  # noqa: F401
        delete_scheduled_job,  # noqa: F401
        delete_space_secret,  # noqa: F401
        delete_space_variable,  # noqa: F401
        delete_space_volumes,  # noqa: F401
        delete_tag,  # noqa: F401
        delete_webhook,  # noqa: F401
        disable_space_dev_mode,  # noqa: F401
        disable_webhook,  # noqa: F401
        download_bucket_files,  # noqa: F401
        duplicate_repo,  # noqa: F401
        edit_discussion_comment,  # noqa: F401
        enable_space_dev_mode,  # noqa: F401
        enable_webhook,  # noqa: F401
        fetch_job_logs,  # noqa: F401
        fetch_job_metrics,  # noqa: F401
        fetch_space_logs,  # noqa: F401
        file_exists,  # noqa: F401
        get_bucket_file_metadata,  # noqa: F401
        get_bucket_paths_info,  # noqa: F401
        get_collection,  # noqa: F401
        get_dataset_leaderboard,  # noqa: F401
        get_dataset_tags,  # noqa: F401
        get_discussion_details,  # noqa: F401
        get_full_repo_name,  # noqa: F401
        get_inference_endpoint,  # noqa: F401
        get_local_safetensors_metadata,  # noqa: F401
        get_model_tags,  # noqa: F401
        get_organization_overview,  # noqa: F401
        get_paths_info,  # noqa: F401
        get_repo_discussions,  # noqa: F401
        get_safetensors_metadata,  # noqa: F401
        get_space_runtime,  # noqa: F401
        get_space_secrets,  # noqa: F401
        get_space_variables,  # noqa: F401
        get_user_overview,  # noqa: F401
        get_webhook,  # noqa: F401
        get_zero_gpu_quota,  # noqa: F401
        grant_access,  # noqa: F401
        inspect_job,  # noqa: F401
        inspect_scheduled_job,  # noqa: F401
        kernel_info,  # noqa: F401
        list_accepted_access_requests,  # noqa: F401
        list_bucket_tree,  # noqa: F401
        list_buckets,  # noqa: F401
        list_collections,  # noqa: F401
        list_daily_papers,  # noqa: F401
        list_dataset_parquet_files,  # noqa: F401
        list_datasets,  # noqa: F401
        list_inference_catalog,  # noqa: F401
        list_inference_endpoints,  # noqa: F401
        list_inference_endpoints_hardware,  # noqa: F401
        list_jobs,  # noqa: F401
        list_jobs_hardware,  # noqa: F401
        list_lfs_files,  # noqa: F401
        list_liked_repos,  # noqa: F401
        list_models,  # noqa: F401
        list_organization_followers,  # noqa: F401
        list_organization_members,  # noqa: F401
        list_papers,  # noqa: F401
        list_pending_access_requests,  # noqa: F401
        list_rejected_access_requests,  # noqa: F401
        list_repo_commits,  # noqa: F401
        list_repo_files,  # noqa: F401
        list_repo_likers,  # noqa: F401
        list_repo_refs,  # noqa: F401
        list_repo_tree,  # noqa: F401
        list_space_templates,  # noqa: F401
        list_spaces,  # noqa: F401
        list_spaces_hardware,  # noqa: F401
        list_user_followers,  # noqa: F401
        list_user_following,  # noqa: F401
        list_user_repos,  # noqa: F401
        list_webhooks,  # noqa: F401
        merge_pull_request,  # noqa: F401
        model_info,  # noqa: F401
        move_bucket,  # noqa: F401
        move_repo,  # noqa: F401
        paper_info,  # noqa: F401
        parse_local_safetensors_file_metadata,  # noqa: F401
        parse_safetensors_file_metadata,  # noqa: F401
        pause_inference_endpoint,  # noqa: F401
        pause_space,  # noqa: F401
        permanently_delete_lfs_files,  # noqa: F401
        preupload_lfs_files,  # noqa: F401
        read_paper,  # noqa: F401
        reject_access_request,  # noqa: F401
        rename_discussion,  # noqa: F401
        repo_exists,  # noqa: F401
        repo_info,  # noqa: F401
        request_space_hardware,  # noqa: F401
        rerun_job,  # noqa: F401
        resolve_revision,  # noqa: F401
        restart_space,  # noqa: F401
        resume_inference_endpoint,  # noqa: F401
        resume_scheduled_job,  # noqa: F401
        revision_exists,  # noqa: F401
        run_as_future,  # noqa: F401
        run_job,  # noqa: F401
        run_uv_job,  # noqa: F401
        scale_to_zero_inference_endpoint,  # noqa: F401
        search_spaces,  # noqa: F401
        set_space_sleep_time,  # noqa: F401
        set_space_volumes,  # noqa: F401
        space_info,  # noqa: F401
        super_squash_history,  # noqa: F401
        suspend_scheduled_job,  # noqa: F401
        sync_bucket,  # noqa: F401
        sync_job_volume,  # noqa: F401
        trigger_scheduled_job,  # noqa: F401
        unlike,  # noqa: F401
        update_bucket_settings,  # noqa: F401
        update_collection_item,  # noqa: F401
        update_collection_metadata,  # noqa: F401
        update_collection_resource_group,  # noqa: F401
        update_inference_endpoint,  # noqa: F401
        update_job_expose,  # noqa: F401
        update_job_labels,  # noqa: F401
        update_repo_settings,  # noqa: F401
        update_scheduled_job_labels,  # noqa: F401
        update_scheduled_job_schedule,  # noqa: F401
        update_webhook,  # noqa: F401
        upload_file,  # noqa: F401
        upload_folder,  # noqa: F401
        verify_repo_checksums,  # noqa: F401
        wait_for_job,  # noqa: F401
        wait_for_space,  # noqa: F401
        whoami,  # noqa: F401
    )
    from .hf_file_system import (
        HfFileSystem,  # noqa: F401
        HfFileSystemFile,  # noqa: F401
        HfFileSystemResolvedPath,  # noqa: F401
        HfFileSystemStreamFile,  # noqa: F401
        hffs,  # noqa: F401
    )
    from .hub_mixin import (
        ModelHubMixin,  # noqa: F401
        PyTorchModelHubMixin,  # noqa: F401
    )
    from .inference._client import (
        InferenceClient,  # noqa: F401
        InferenceTimeoutError,  # noqa: F401
    )
    from .inference._generated._async_client import AsyncInferenceClient  # noqa: F401
    from .inference._generated.types import (
        AudioClassificationInput,  # noqa: F401
        AudioClassificationOutputElement,  # noqa: F401
        AudioClassificationOutputTransform,  # noqa: F401
        AudioClassificationParameters,  # noqa: F401
        AudioToAudioInput,  # noqa: F401
        AudioToAudioOutputElement,  # noqa: F401
        AutomaticSpeechRecognitionEarlyStoppingEnum,  # noqa: F401
        AutomaticSpeechRecognitionGenerationParameters,  # noqa: F401
        AutomaticSpeechRecognitionInput,  # noqa: F401
        AutomaticSpeechRecognitionOutput,  # noqa: F401
        AutomaticSpeechRecognitionOutputChunk,  # noqa: F401
        AutomaticSpeechRecognitionParameters,  # noqa: F401
        ChatCompletionInput,  # noqa: F401
        ChatCompletionInputFunctionDefinition,  # noqa: F401
        ChatCompletionInputFunctionName,  # noqa: F401
        ChatCompletionInputGrammarType,  # noqa: F401
        ChatCompletionInputJSONSchema,  # noqa: F401
        ChatCompletionInputMessage,  # noqa: F401
        ChatCompletionInputMessageChunk,  # noqa: F401
        ChatCompletionInputMessageChunkType,  # noqa: F401
        ChatCompletionInputResponseFormatJSONObject,  # noqa: F401
        ChatCompletionInputResponseFormatJSONSchema,  # noqa: F401
        ChatCompletionInputResponseFormatText,  # noqa: F401
        ChatCompletionInputStreamOptions,  # noqa: F401
        ChatCompletionInputTool,  # noqa: F401
        ChatCompletionInputToolCall,  # noqa: F401
        ChatCompletionInputToolChoiceClass,  # noqa: F401
        ChatCompletionInputToolChoiceEnum,  # noqa: F401
        ChatCompletionInputURL,  # noqa: F401
        ChatCompletionOutput,  # noqa: F401
        ChatCompletionOutputComplete,  # noqa: F401
        ChatCompletionOutputFunctionDefinition,  # noqa: F401
        ChatCompletionOutputLogprob,  # noqa: F401
        ChatCompletionOutputLogprobs,  # noqa: F401
        ChatCompletionOutputMessage,  # noqa: F401
        ChatCompletionOutputToolCall,  # noqa: F401
        ChatCompletionOutputTopLogprob,  # noqa: F401
        ChatCompletionOutputUsage,  # noqa: F401
        ChatCompletionStreamOutput,  # noqa: F401
        ChatCompletionStreamOutputChoice,  # noqa: F401
        ChatCompletionStreamOutputDelta,  # noqa: F401
        ChatCompletionStreamOutputDeltaToolCall,  # noqa: F401
        ChatCompletionStreamOutputFunction,  # noqa: F401
        ChatCompletionStreamOutputLogprob,  # noqa: F401
        ChatCompletionStreamOutputLogprobs,  # noqa: F401
        ChatCompletionStreamOutputTopLogprob,  # noqa: F401
        ChatCompletionStreamOutputUsage,  # noqa: F401
        DepthEstimationInput,  # noqa: F401
        DepthEstimationOutput,  # noqa: F401
        DocumentQuestionAnsweringInput,  # noqa: F401
        DocumentQuestionAnsweringInputData,  # noqa: F401
        DocumentQuestionAnsweringOutputElement,  # noqa: F401
        DocumentQuestionAnsweringParameters,  # noqa: F401
        FeatureExtractionInput,  # noqa: F401
        FeatureExtractionInputTruncationDirection,  # noqa: F401
        FillMaskInput,  # noqa: F401
        FillMaskOutputElement,  # noqa: F401
        FillMaskParameters,  # noqa: F401
        ImageClassificationInput,  # noqa: F401
        ImageClassificationOutputElement,  # noqa: F401
        ImageClassificationOutputTransform,  # noqa: F401
        ImageClassificationParameters,  # noqa: F401
        ImageSegmentationInput,  # noqa: F401
        ImageSegmentationOutputElement,  # noqa: F401
        ImageSegmentationParameters,  # noqa: F401
        ImageSegmentationSubtask,  # noqa: F401
        ImageTextToImageInput,  # noqa: F401
        ImageTextToImageOutput,  # noqa: F401
        ImageTextToImageParameters,  # noqa: F401
        ImageTextToImageTargetSize,  # noqa: F401
        ImageTextToVideoInput,  # noqa: F401
        ImageTextToVideoOutput,  # noqa: F401
        ImageTextToVideoParameters,  # noqa: F401
        ImageTextToVideoTargetSize,  # noqa: F401
        ImageToImageInput,  # noqa: F401
        ImageToImageOutput,  # noqa: F401
        ImageToImageParameters,  # noqa: F401
        ImageToImageTargetSize,  # noqa: F401
        ImageToTextEarlyStoppingEnum,  # noqa: F401
        ImageToTextGenerationParameters,  # noqa: F401
        ImageToTextInput,  # noqa: F401
        ImageToTextOutput,  # noqa: F401
        ImageToTextParameters,  # noqa: F401
        ImageToVideoInput,  # noqa: F401
        ImageToVideoOutput,  # noqa: F401
        ImageToVideoParameters,  # noqa: F401
        ImageToVideoTargetSize,  # noqa: F401
        ObjectDetectionBoundingBox,  # noqa: F401
        ObjectDetectionInput,  # noqa: F401
        ObjectDetectionOutputElement,  # noqa: F401
        ObjectDetectionParameters,  # noqa: F401
        Padding,  # noqa: F401
        QuestionAnsweringInput,  # noqa: F401
        QuestionAnsweringInputData,  # noqa: F401
        QuestionAnsweringOutputElement,  # noqa: F401
        QuestionAnsweringParameters,  # noqa: F401
        SentenceSimilarityInput,  # noqa: F401
        SentenceSimilarityInputData,  # noqa: F401
        SummarizationInput,  # noqa: F401
        SummarizationOutput,  # noqa: F401
        SummarizationParameters,  # noqa: F401
        SummarizationTruncationStrategy,  # noqa: F401
        TableQuestionAnsweringInput,  # noqa: F401
        TableQuestionAnsweringInputData,  # noqa: F401
        TableQuestionAnsweringOutputElement,  # noqa: F401
        TableQuestionAnsweringParameters,  # noqa: F401
        Text2TextGenerationInput,  # noqa: F401
        Text2TextGenerationOutput,  # noqa: F401
        Text2TextGenerationParameters,  # noqa: F401
        Text2TextGenerationTruncationStrategy,  # noqa: F401
        TextClassificationInput,  # noqa: F401
        TextClassificationOutputElement,  # noqa: F401
        TextClassificationOutputTransform,  # noqa: F401
        TextClassificationParameters,  # noqa: F401
        TextGenerationInput,  # noqa: F401
        TextGenerationInputGenerateParameters,  # noqa: F401
        TextGenerationInputGrammarType,  # noqa: F401
        TextGenerationOutput,  # noqa: F401
        TextGenerationOutputBestOfSequence,  # noqa: F401
        TextGenerationOutputDetails,  # noqa: F401
        TextGenerationOutputFinishReason,  # noqa: F401
        TextGenerationOutputPrefillToken,  # noqa: F401
        TextGenerationOutputToken,  # noqa: F401
        TextGenerationStreamOutput,  # noqa: F401
        TextGenerationStreamOutputStreamDetails,  # noqa: F401
        TextGenerationStreamOutputToken,  # noqa: F401
        TextToAudioEarlyStoppingEnum,  # noqa: F401
        TextToAudioGenerationParameters,  # noqa: F401
        TextToAudioInput,  # noqa: F401
        TextToAudioOutput,  # noqa: F401
        TextToAudioParameters,  # noqa: F401
        TextToImageInput,  # noqa: F401
        TextToImageOutput,  # noqa: F401
        TextToImageParameters,  # noqa: F401
        TextToSpeechEarlyStoppingEnum,  # noqa: F401
        TextToSpeechGenerationParameters,  # noqa: F401
        TextToSpeechInput,  # noqa: F401
        TextToSpeechOutput,  # noqa: F401
        TextToSpeechParameters,  # noqa: F401
        TextToVideoInput,  # noqa: F401
        TextToVideoOutput,  # noqa: F401
        TextToVideoParameters,  # noqa: F401
        TokenClassificationAggregationStrategy,  # noqa: F401
        TokenClassificationInput,  # noqa: F401
        TokenClassificationOutputElement,  # noqa: F401
        TokenClassificationParameters,  # noqa: F401
        TranslationInput,  # noqa: F401
        TranslationOutput,  # noqa: F401
        TranslationParameters,  # noqa: F401
        TranslationTruncationStrategy,  # noqa: F401
        TypeEnum,  # noqa: F401
        VideoClassificationInput,  # noqa: F401
        VideoClassificationOutputElement,  # noqa: F401
        VideoClassificationOutputTransform,  # noqa: F401
        VideoClassificationParameters,  # noqa: F401
        VisualQuestionAnsweringInput,  # noqa: F401
        VisualQuestionAnsweringInputData,  # noqa: F401
        VisualQuestionAnsweringOutputElement,  # noqa: F401
        VisualQuestionAnsweringParameters,  # noqa: F401
        ZeroShotClassificationInput,  # noqa: F401
        ZeroShotClassificationOutputElement,  # noqa: F401
        ZeroShotClassificationParameters,  # noqa: F401
        ZeroShotImageClassificationInput,  # noqa: F401
        ZeroShotImageClassificationOutputElement,  # noqa: F401
        ZeroShotImageClassificationParameters,  # noqa: F401
        ZeroShotObjectDetectionBoundingBox,  # noqa: F401
        ZeroShotObjectDetectionInput,  # noqa: F401
        ZeroShotObjectDetectionOutputElement,  # noqa: F401
        ZeroShotObjectDetectionParameters,  # noqa: F401
    )
    from .inference._mcp.agent import Agent  # noqa: F401
    from .inference._mcp.mcp_client import MCPClient  # noqa: F401
    from .repocard import (
        DatasetCard,  # noqa: F401
        ModelCard,  # noqa: F401
        RepoCard,  # noqa: F401
        SpaceCard,  # noqa: F401
        metadata_eval_result,  # noqa: F401
        metadata_load,  # noqa: F401
        metadata_save,  # noqa: F401
        metadata_update,  # noqa: F401
    )
    from .repocard_data import (
        CardData,  # noqa: F401
        DatasetCardData,  # noqa: F401
        EvalResult,  # noqa: F401
        ModelCardData,  # noqa: F401
        SpaceCardData,  # noqa: F401
    )
    from .serialization import (
        StateDictSplit,  # noqa: F401
        get_torch_storage_id,  # noqa: F401
        get_torch_storage_size,  # noqa: F401
        load_state_dict_from_file,  # noqa: F401
        load_torch_model,  # noqa: F401
        save_torch_model,  # noqa: F401
        save_torch_state_dict,  # noqa: F401
        split_state_dict_into_shards_factory,  # noqa: F401
        split_torch_state_dict_into_shards,  # noqa: F401
    )
    from .serialization._dduf import (
        DDUFEntry,  # noqa: F401
        export_entries_as_dduf,  # noqa: F401
        export_folder_as_dduf,  # noqa: F401
        read_dduf_file,  # noqa: F401
    )
    from .utils import (
        ASYNC_CLIENT_FACTORY_T,  # noqa: F401
        CLIENT_FACTORY_T,  # noqa: F401
        CachedFileInfo,  # noqa: F401
        CachedIncompleteFileInfo,  # noqa: F401
        CachedRepoInfo,  # noqa: F401
        CachedRevisionInfo,  # noqa: F401
        CacheNotFound,  # noqa: F401
        CorruptedCacheException,  # noqa: F401
        DeleteCacheStrategy,  # noqa: F401
        HFCacheInfo,  # noqa: F401
        HfUri,  # noqa: F401
        cached_assets_path,  # noqa: F401
        close_session,  # noqa: F401
        dump_environment_info,  # noqa: F401
        get_async_session,  # noqa: F401
        get_session,  # noqa: F401
        get_token,  # noqa: F401
        hf_raise_for_status,  # noqa: F401
        logging,  # noqa: F401
        parse_hf_uri,  # noqa: F401
        scan_cache_dir,  # noqa: F401
        set_async_client_factory,  # noqa: F401
        set_client_factory,  # noqa: F401
    )
<|endoftext|># coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
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
"""Streamed, multi-commit upload of a folder on top of the Xet upload protocol.

How it works:

- The **coordinator** (caller's thread) walks the list of files and asks the Hub, 256 files at a
  time, what each file is (regular git blob, xet file, ignored). Regular files are accumulated
  directly; xet files are registered into a `XetSession` upload-commit, which chunks, deduplicates,
  retries and uploads them in the background while the coordinator keeps going. No Python-side
  sha256 computation: `hf_xet` computes it during chunking (single read pass over each file).
- Whenever enough files have accumulated (adaptive batch size), the batch is handed over to the
  **committer** thread which joins the xet uploads, drops unchanged files, and creates a git
  commit for the batch. While a batch is being committed, the coordinator is already uploading
  the next one.
- Interrupted uploads are resumable by simply re-running the same call: already-committed files
  are dropped (no-op detection against the remote oid) and already-uploaded chunks are
  deduplicated by the xet storage backend, transferring ~0 bytes.
"""

import queue
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import quote

from . import constants
from ._commit_api import (
    CommitOperationAdd,
    CommitOperationDelete,
    _fetch_upload_modes,
    _send_commit,
    _warn_on_overwriting_operations,
)
from .errors import RepositoryNotFoundError
from .utils import are_progress_bars_disabled, logging
from .utils._xet import (
    XetTokenType,
    abort_xet_session,
    get_xet_session,
    xet_connection_info_refresh_url,
    xet_headers_without_auth,
)


if TYPE_CHECKING:
    from .hf_api import CommitInfo, HfApi

logger = logging.get_logger(__name__)

# Number of files sent to the "preupload" endpoint per call (server-side limit).
PREUPLOAD_BATCH_SIZE = 256

# Files per git commit: adaptive, scaled up after fast commits and down after failures.
COMMIT_SIZE_SCALE = [20, 50, 75, 100, 125, 200, 250, 400, 600, 1000]
INITIAL_COMMIT_SIZE_INDEX = 6  # start at 256 files per commit
TARGET_COMMIT_DURATION = 40.0  # seconds; scale up batch size if commits are faster than this
MAX_COMMIT_INTERVAL = 5 * 60.0  # seconds; force a commit if the current batch is older than this

# Budget of regular-file content per commit (regular files are base64-encoded in the payload).
REGULAR_CONTENT_BYTES_BUDGET = 100 * 1024 * 1024

_SENTINEL = object()  # Sentinel value for the batch queue to indicate the end of the upload

# Live display tuning
_BAR_WIDTH = 20
_REFRESH_INTERVAL = 0.5  # seconds between redraws on a TTY
_NON_TTY_LOG_INTERVAL = 30.0  # seconds between summary logs when stderr is not a TTY


def _bar(current: float, total: float, width: int = _BAR_WIDTH) -> str:
    if total <= 0:
        return "░" * width
    filled = int(min(current / total, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _format_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000:
            if n < 10:
                return f"{n:.2f}{unit}"
            elif n < 100:
                return f"{n:.1f}{unit}"
            return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.1f}PB"


class _LiveDisplay:
    """Live progress display on stderr::

        Preparing   ████████████████████  11,100 / 11,100 ✓
        Uploading   ██████████████░░░░░░  580 / 603 files  3.8GB · 19.7MB/s
        Validating  ████████░░░░░░░░░░░░  43%
        Committing  ██████████████████░░  10,800 / 11,100  14 commits

    The Validating line appears once the v2 shard finalization API starts reporting
    progress. Shard upload / validation / sync are folded into this single bar so
    users are not exposed to Xet-internal "shard" jargon (and so there is only one
    Committing line — the git commit).

    A small renderer thread redraws all lines in-place every ~0.5 s on a TTY
    (worker threads only update counters under a lock). When stderr is not a TTY,
    it falls back to a periodic ``logger.info`` summary instead.

    Disabling progress bars (e.g. agent output mode) only turns off the TTY renderer:
    the non-TTY log summaries are gated by the logger verbosity alone, so consumers
    tailing stderr during a long upload still see periodic progress.
    """

    def __init__(self, total_files: int, enabled: bool = True) -> None:
        self._total = total_files
        self._tty = enabled and sys.stderr.isatty()
        self._active = self._tty or logger.isEnabledFor(logging.INFO)
        self._lock = threading.Lock()
        self._drawn_lines = 0  # number of lines printed in the last _redraw call
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Counters (written by coordinator/committer threads, read by the renderer)
        self._prepared = 0
        self._ignored = 0
        self._xet_total = 0
        self._xet_done: set[str] = set()  # item names; unique across batches
        self._committed = 0  # committed or skipped-as-unchanged
        self._nb_commits = 0

        # Xet transfer bytes, summed across (possibly concurrent) upload-commits
        self._xet_bytes = 0
        self._speed_ema = 0.0
        self._prev_bytes = 0
        self._prev_time: float | None = None

        # Shard finalization → single Validating bar (v2 API; activated on first event)
        self._shard_active = False
        self._shard_val_total = 0
        self._shard_val_done = 0
        self._shards_total = 0
        self._shards_done = 0

    # -- lifecycle (main thread) ------------------------------------------------

    def start(self) -> None:
        if not self._active:
            return
        if self._tty:
            sys.stderr.write(f"Found {self._total:,} files to upload\n")
            sys.stderr.flush()
        else:
            logger.info(f"Found {self._total:,} files to upload")
        self._thread = threading.Thread(target=self._render_loop, name="hf-upload-display", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
        if self._tty:
            with self._lock:
                self._redraw()  # final state

    # -- counter updates (coordinator / committer / xet callback threads) --------

    def notify_prepared(self, n: int) -> None:
        with self._lock:
            self._prepared += n

    def notify_ignored(self, n: int) -> None:
        with self._lock:
            self._ignored += n

    def notify_xet_registered(self, n: int) -> None:
        with self._lock:
            self._xet_total += n

    def notify_xet_uploaded(self, names: list[str]) -> None:
        with self._lock:
            self._xet_done.update(names)

    def notify_skipped(self, n: int) -> None:
        with self._lock:
            self._committed += n

    def notify_commit(self, n_files: int) -> None:
        with self._lock:
            self._committed += n_files
            self._nb_commits += 1

    def new_xet_callback(self) -> "Callable | None":
        """Progress callback for one ``new_upload_commit``.

        The byte counters in ``group_report`` are cumulative *per upload-commit* and several
        upload-commits can be in flight at once (one finalizing, one filling), so each commit
        gets its own closure tracking its own previous value; increments are summed globally.
        """
        if not self._active:
            return None
        prev_transfer = 0
        prev_shard_val_done = 0
        prev_shard_val_total = 0
        prev_shards_done = 0
        prev_shards_total = 0

        def callback(group_report: Any, item_reports: Any) -> None:
            nonlocal prev_transfer
            nonlocal prev_shard_val_done, prev_shard_val_total
            nonlocal prev_shards_done, prev_shards_total
            with self._lock:
                completed = group_report.total_transfer_bytes_completed
                self._xet_bytes += max(0, completed - prev_transfer)
                prev_transfer = completed
                for item in item_reports.values():
                    if item.total_bytes > 0 and item.bytes_completed == item.total_bytes:
                        self._xet_done.add(item.item_name)

                shard = getattr(group_report, "shard", None)
                if shard is not None and shard.total_shards > 0:
                    self._shard_active = True

                    v = shard.total_shard_validation_entries_completed
                    self._shard_val_done += max(0, v - prev_shard_val_done)
                    prev_shard_val_done = v

                    v = shard.total_shard_validation_entries
                    self._shard_val_total += max(0, v - prev_shard_val_total)
                    prev_shard_val_total = v

                    v = shard.total_shards_completed
                    self._shards_done += max(0, v - prev_shards_done)
                    prev_shards_done = v

                    v = shard.total_shards
                    self._shards_total += max(0, v - prev_shards_total)
                    prev_shards_total = v

        return callback

    # -- rendering (display thread) ----------------------------------------------

    def _render_loop(self) -> None:
        last_log = 0.0
        while not self._stop_event.wait(_REFRESH_INTERVAL):
            with self._lock:
                self._update_speed()
                if self._tty:
                    self._redraw()
                elif time.monotonic() - last_log >= _NON_TTY_LOG_INTERVAL:
                    logger.info(self._summary())
                    last_log = time.monotonic()

    def _update_speed(self) -> None:
        now = time.monotonic()
        if self._prev_time is not None and now > self._prev_time:
            rate = (self._xet_bytes - self._prev_bytes) / (now - self._prev_time)
            self._speed_ema = rate if self._speed_ema == 0 else 0.3 * rate + 0.7 * self._speed_ema
        self._prev_time = now
        self._prev_bytes = self._xet_bytes

    def _redraw(self) -> None:
        if self._drawn_lines > 0:
            sys.stderr.write(f"\033[{self._drawn_lines}A")
        # Validating sits between Uploading and Committing (before the git commit bar).
        lines = [self._line_preparing(), self._line_uploading()]
        if self._shard_active:
            lines.append(self._line_validating())
        lines.append(self._line_committing())
        width = shutil.get_terminal_size().columns
        for line in lines:
            truncated = line[: width - 4] + "..." if len(line) > width - 1 else line
            sys.stderr.write(f"\r\033[K{truncated}\n")
        sys.stderr.flush()
        self._drawn_lines = len(lines)

    def _line_preparing(self) -> str:
        done = " ✓" if self._prepared >= self._total else ""
        return f"  Preparing   {_bar(self._prepared, self._total)}  {self._prepared:,} / {self._total:,}{done}"

    def _line_uploading(self) -> str:
        if self._xet_total == 0:
            bar = _bar(1, 1) if self._prepared >= self._total else _bar(0, 1)
            return f"  Uploading   {bar}  -"
        n_done = len(self._xet_done)
        parts = []
        if self._xet_bytes > 0:
            parts.append(_format_bytes(self._xet_bytes))
        if self._speed_ema > 0:
            parts.append(f"{_format_bytes(self._speed_ema)}/s")
        extra = f"  {' · '.join(parts)}" if parts else ""
        done = " ✓" if self._prepared >= self._total and n_done >= self._xet_total else ""
        return f"  Uploading   {_bar(n_done, self._xet_total)}  {n_done:,} / {self._xet_total:,} files{extra}{done}"

    def _line_validating(self) -> str:
        # Prefer validation-entry progress (the long phase). Fall back to shard-completion
        # counts when entries are absent (e.g. V1 synthetic Result with no Validating frames).
        # Show a percentage — raw entry counts are opaque Xet internals.
        if self._shard_val_total > 0:
            done_n, total_n = self._shard_val_done, self._shard_val_total
        else:
            done_n, total_n = self._shards_done, self._shards_total
        pct = min(100, round(100 * done_n / total_n)) if total_n > 0 else 0
        return f"  Validating  {_bar(done_n, total_n)}  {pct}%"

    def _line_committing(self) -> str:
        effective = self._total - self._ignored
        commits_str = f"  {self._nb_commits} commits" if self._nb_commits > 1 else ""
        done = " ✓" if self._committed >= effective > 0 else ""
        return (
            f"  Committing  {_bar(self._committed, effective)}  {self._committed:,} / {effective:,}{commits_str}{done}"
        )

    def _summary(self) -> str:
        summary = (
            f"Uploading... {self._prepared:,}/{self._total:,} files checked, "
            f"{len(self._xet_done):,}/{self._xet_total:,} uploaded ({_format_bytes(self._xet_bytes)} transferred), "
            f"{self._committed:,} committed in {self._nb_commits} commit(s)"
        )
        if self._shard_active:
            if self._shard_val_total > 0:
                pct = min(100, round(100 * self._shard_val_done / self._shard_val_total))
                summary += f", validating {pct}%"
            elif self._shards_total > 0:
                pct = min(100, round(100 * self._shards_done / self._shards_total))
                summary += f", validating {pct}%"
        return summary


class _CommitPacer:
    """Adaptive number of files per commit, to stay below server-side commit timeouts."""

    def __init__(self) -> None:
        self._index = INITIAL_COMMIT_SIZE_INDEX

    @property
    def target(self) -> int:
        return COMMIT_SIZE_SCALE[self._index]

    def record_success(self, duration: float, nb_files: int) -> None:
        if duration < TARGET_COMMIT_DURATION and nb_files >= self.target:
            self._index = min(self._index + 1, len(COMMIT_SIZE_SCALE) - 1)
        elif duration > TARGET_COMMIT_DURATION:
            self._index = max(self._index - 1, 0)

    def record_failure(self) -> None:
        self._index = max(self._index - 1, 0)


class _Batch:
    """A group of files destined to a single git commit, with their in-flight xet uploads."""

    def __init__(self) -> None:
        self.ops: list[CommitOperationAdd] = []
        self.regular_bytes: int = 0
        self.xet_commit: Any = None  # XetUploadCommit, opened lazily
        self.handles: list[tuple[CommitOperationAdd, Any]] = []  # (op, XetFileUpload)
        self.created_at: float = time.monotonic()


class _UploadPipeline:
    def __init__(
        self,
        api: "HfApi",
        *,
        repo_id: str,
        repo_type: str,
        add_operations: list[CommitOperationAdd],
        delete_operations: list[CommitOperationDelete],
        commit_message: str,
        commit_description: str | None,
        token: str | bool | None,
        revision: str | None,
        create_pr: bool,
        parent_commit: str | None,
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.add_operations = add_operations
        self.delete_operations = delete_operations
        self.commit_message = commit_message
        self.commit_description = commit_description
        self.token = token
        self.headers = api._build_hf_headers(token=token)
        self.revision = revision or constants.DEFAULT_REVISION
        self.create_pr = create_pr
        self.parent_commit = parent_commit

        # The base revision is used by the coordinator for ALL preupload calls and the xet token
        # refresh URL, with the `create_pr` flag — exactly like `create_commit` does. It never
        # changes during the run, even after a PR has been created.
        self.base_revision_quoted = quote(self.revision, safe="")

        # Committer state (mutated by the committer thread only)
        self.commit_revision_quoted = self.base_revision_quoted  # switched to the PR ref once created
        self.pr_url: str | None = None
        self.pr_revision: str | None = None
        self.nb_commits = 0
        self.last_commit_info: "CommitInfo | None" = None
        self.pacer = _CommitPacer()

        # Pipeline plumbing
        self.batch_queue: queue.Queue = queue.Queue(maxsize=1)
        self.errors: list[BaseException] = []
        self.abort_event = threading.Event()
        self.display = _LiveDisplay(total_files=len(add_operations), enabled=not are_progress_bars_disabled())

        # All xet uploads share the same token refresh URL. With `create_pr`, the final ref is not
        # known in advance: `?create_pr=1` makes the server grant a token valid for PR refs.
        refresh_url = xet_connection_info_refresh_url(
            token_type=XetTokenType.WRITE,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=self.base_revision_quoted,
            endpoint=api.endpoint,
        )
        if create_pr:
            refresh_url += "?create_pr=1"
        self.xet_session = get_xet_session()
        self.xet_commit_kwargs = {
            "token_refresh_url": refresh_url,
            "token_refresh_headers": self.headers,
            "custom_headers": xet_headers_without_auth(self.headers),
        }

        # `.gitignore` rules are enforced server-side: forward the local one if it's being uploaded.
        self.gitignore_content: str | None = None
        for op in add_operations:
            if op.path_in_repo == ".gitignore":
                with op.as_file() as f:
                    self.gitignore_content = f.read().decode()
                break

    def run(self) -> "CommitInfo":
        _warn_on_overwriting_operations([*self.delete_operations, *self.add_operations])
        committer = threading.Thread(target=self._committer_loop, name="hf-upload-committer", daemon=True)
        committer.start()
        self.display.start()
        try:
            self._coordinator_loop()
        except BaseException:
            self.abort_event.set()
            abort_xet_session()
            raise
        finally:
            if self.abort_event.is_set():
                # The committer exits on its own once the queue is drained (see `_committer_loop`).
                # Bound the wait so a xet call blocked on the (aborted) session can never hang the
                # shutdown — the committer is a daemon thread.
                committer.join(timeout=10)
            else:
                self.batch_queue.put(_SENTINEL)
                committer.join()
            self.display.close()
            if self.abort_event.is_set() and self.pr_revision is not None:
                logger.warning(
                    f"Upload to pull request {self.pr_url} did not complete. To resume into the"
                    f' same PR, re-run with `revision="{self.pr_revision}"` (without `create_pr=True`). Re-running'
                    " with `create_pr=True` would open a new pull request."
                )
        if self.errors:
            raise self.errors[0]
        return self._final_commit_info()

    # ---------------------------------------------------------------- coordinator

    def _coordinator_loop(self) -> None:
        import hf_xet

        batch = _Batch()
        for start in range(0, len(self.add_operations), PREUPLOAD_BATCH_SIZE):
            if self.abort_event.is_set():
                self._abort_batch(batch)
                return
            chunk = self.add_operations[start : start + PREUPLOAD_BATCH_SIZE]
            try:
                _fetch_upload_modes(
                    additions=chunk,
                    repo_type=self.repo_type,
                    repo_id=self.repo_id,
                    headers=self.headers,
                    revision=self.base_revision_quoted,
                    endpoint=self.api.endpoint,
                    create_pr=self.create_pr,
                    gitignore_content=self.gitignore_content,
                )
            except RepositoryNotFoundError as e:
                from .hf_api import _CREATE_COMMIT_NO_REPO_ERROR_MESSAGE

                e.append_to_message(_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE)
                raise
            self.display.notify_prepared(len(chunk))
            for op in chunk:
                if op._should_ignore:
                    logger.debug(f"Skipping upload for '{op.path_in_repo}' (ignored by gitignore rules).")
                    self.display.notify_ignored(1)
                    continue
                if op._upload_mode == "regular":
                    batch.regular_bytes += op.upload_info.size
                else:
                    if batch.xet_commit is None:
                        batch.xet_commit = self.xet_session.new_upload_commit(
                            progress_callback=self.display.new_xet_callback(), **self.xet_commit_kwargs
                        )
                    # Upload starts immediately in the background. sha256 is computed by hf_xet
                    # while chunking, unless already known (e.g. resumed operations).
                    sha_arg = op.upload_info.sha256.hex() if op.upload_info.is_hashed else hf_xet.COMPUTE_SHA256
                    if isinstance(op.path_or_fileobj, bytes):
                        handle = batch.xet_commit.start_upload_bytes(
                            op.path_or_fileobj, sha256=sha_arg, name=op.path_in_repo
                        )
                    else:
                        handle = batch.xet_commit.start_upload_file(str(op.path_or_fileobj), sha256=sha_arg)
                    batch.handles.append((op, handle))
                    self.display.notify_xet_registered(1)
                batch.ops.append(op)

                if (
                    len(batch.ops) >= self.pacer.target
                    or batch.regular_bytes >= REGULAR_CONTENT_BYTES_BUDGET
                    or (time.monotonic() - batch.created_at > MAX_COMMIT_INTERVAL and len(batch.ops) > 0)
                ):
                    self._enqueue(batch)
                    batch = _Batch()
        self._enqueue(batch)

    def _enqueue(self, batch: _Batch) -> None:
        if len(batch.ops) == 0 and not (self.nb_commits == 0 and len(self.delete_operations) > 0):
            return
        # Blocks if a batch is already waiting: natural backpressure on scanning/uploading.
        while not self.abort_event.is_set():
            try:
                self.batch_queue.put(batch, timeout=1.0)
                return
            except queue.Full:
                continue
        self._abort_batch(batch)

    def _abort_batch(self, batch: _Batch) -> None:
        if batch.xet_commit is not None:
            try:
                batch.xet_commit.abort()
            except Exception:
                pass

    # ---------------------------------------------------------------- committer

    def _committer_loop(self) -> None:
        while True:
            try:
                batch = self.batch_queue.get(timeout=0.5)
            except queue.Empty:
                if self.abort_event.is_set():
                    return  # aborted: exit once the queue is drained, no sentinel needed
                continue
            if batch is _SENTINEL:
                return
            try:
                if not self.abort_event.is_set():
                    self._process_batch(batch)
                else:
                    self._abort_batch(batch)
            except BaseException as e:
                self._abort_batch(batch)
                self.errors.append(e)
                self.abort_event.set()

    def _process_batch(self, batch: _Batch) -> None:
        # 1. Wait for all xet uploads of this batch and finalize them (atomic xet commit). Files
        #    can only be referenced by a git commit once their xet upload-commit is finalized.
        if batch.xet_commit is not None:
            batch.xet_commit.wait_to_finish()
            for op, handle in batch.handles:
                if not op.upload_info.is_hashed:
                    op.upload_info.sha256 = bytes.fromhex(handle.result().xet_info.sha256)
                op._is_uploaded = True
            # Files whose last progress tick was missed are still done at this point.
            self.display.notify_xet_uploaded(
                [
                    str(op.path_or_fileobj) if not isinstance(op.path_or_fileobj, bytes) else op.path_in_repo
                    for op, _ in batch.handles
                ]
            )

        # 2. Drop files that have not changed compared to the remote (prevents empty commits).
        #    Their chunks were deduplicated anyway (~0 bytes transferred).
        ops_to_commit = []
        for op in batch.ops:
            if op._remote_oid is not None and op._remote_oid == op._local_oid:
                logger.debug(f"Skipping commit for '{op.path_in_repo}' (file unchanged).")
                self.display.notify_skipped(1)
                continue
            ops_to_commit.append(op)

        # 3. Create the git commit(s). On failure, scale down and split the batch.
        if len(ops_to_commit) > 0 or (self.nb_commits == 0 and len(self.delete_operations) > 0):
            self._commit_with_split(ops_to_commit)

    def _commit_with_split(self, ops: list[CommitOperationAdd]) -> None:
        try:
            self._do_commit(ops)
        except Exception as e:
            self.pacer.record_failure()
            if len(ops) <= COMMIT_SIZE_SCALE[0]:
                raise
            logger.warning(f"Commit of {len(ops)} files failed ({e!r}). Retrying in smaller chunks.")
            target = self.pacer.target
            for start in range(0, len(ops), target):
                self._commit_with_split(ops[start : start + target])

    def _do_commit(self, ops: list[CommitOperationAdd]) -> None:
        if self.create_pr and self.pr_revision is None:
            # Create the (draft) pull request explicitly and push every commit to its ref. Committing
            # with `?create_pr=1` instead would risk opening a second PR if the commit POST is retried
            # after a lost response. Created lazily so that a fully-unchanged upload opens no PR.
            # Note: PRs created this way are always opened against the default branch, hence the
            # `create_pr` + `revision` combination being rejected in `upload_folder`.
            pr = self.api.create_pull_request(
                repo_id=self.repo_id,
                title=self.commit_message,
                token=self.token,
                description=self.commit_description,
                repo_type=self.repo_type,
            )
            if pr.git_reference is None:
                raise ValueError("Server did not return a git reference for the created pull request.")
            self.pr_url = pr.url
            self.pr_revision = pr.git_reference
            self.commit_revision_quoted = quote(pr.git_reference, safe="")

        operations: list[Any] = list(ops)
        if self.nb_commits == 0:
            # Deletions and `parent_commit` ride the first commit.
            operations = list(self.delete_operations) + operations

        commit_message = (
            self.commit_message if self.nb_commits == 0 else f"{self.commit_message} (part {self.nb_commits + 1})"
        )
        t0 = time.monotonic()
        # Retried with backoff on transient errors: safe because the commit targets an explicit
        # ref (`?create_pr=1` is never used, see above).
        self.last_commit_info = _send_commit(
            operations=operations,
            files_to_copy={},
            commit_message=commit_message,
            commit_description=self.commit_description or "",
            repo_type=self.repo_type,
            repo_id=self.repo_id,
            headers=self.headers,
            revision=self.commit_revision_quoted,
            endpoint=self.api.endpoint,
            parent_commit=self.parent_commit if self.nb_commits == 0 else None,
            retry_on_error=True,
        )
        duration = time.monotonic() - t0
        self.pacer.record_success(duration, len(ops))
        self.nb_commits += 1

        for op in ops:
            op._is_committed = True
        self.display.notify_commit(len(ops))
        logger.debug(f"Committed {len(ops)} file(s) in {duration:.1f}s: {self.last_commit_info.commit_url}")

    # ---------------------------------------------------------------- result

    def _final_commit_info(self) -> "CommitInfo":
        from .hf_api import CommitInfo

        if self.last_commit_info is None:
            # Nothing was committed (everything unchanged/ignored): mimic `create_commit` and
            # return info about the latest commit on the target revision.
            logger.warning("No files have been modified since last commit. Skipping to prevent empty commit.")
            info = self.api.repo_info(
                repo_id=self.repo_id, repo_type=self.repo_type, revision=self.revision, token=self.token
            )
            url_prefix = self.api.endpoint
            if self.repo_type != constants.REPO_TYPE_MODEL:
                url_prefix = f"{url_prefix}/{self.repo_type}s"
            return CommitInfo(
                commit_url=f"{url_prefix}/{self.repo_id}/commit/{info.sha}",
                commit_message=self.commit_message,
                commit_description=self.commit_description or "",
                oid=info.sha,  # type: ignore
                _endpoint=self.api.endpoint,
            )
        if self.nb_commits > 1:
            logger.info(f"Upload completed in {self.nb_commits} commits.")
        if self.pr_url is not None:
            # PR upload: attach the PR info (commit responses don't carry it; the PR is created separately).
            return CommitInfo(
                commit_url=self.last_commit_info.commit_url,
                commit_message=self.last_commit_info.commit_message,
                commit_description=self.last_commit_info.commit_description,
                oid=self.last_commit_info.oid,
                pr_url=self.pr_url,
                _endpoint=self.api.endpoint,
            )
        return self.last_commit_info


def pipelined_upload(
    api: "HfApi",
    *,
    repo_id: str,
    repo_type: str,
    add_operations: list[CommitOperationAdd],
    delete_operations: list[CommitOperationDelete],
    commit_message: str,
    commit_description: str | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    create_pr: bool = False,
    parent_commit: str | None = None,
) -> "CommitInfo":
    """Upload a prepared list of operations through the streamed multi-commit pipeline.

    Requires `hf_xet` to be installed. See module docstring for the architecture.
    """

    return _UploadPipeline(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        add_operations=add_operations,
        delete_operations=delete_operations,
        commit_message=commit_message,
        commit_description=commit_description,
        token=token,
        revision=revision,
        create_pr=create_pr,
        parent_commit=parent_commit,
    ).run()
<|endoftext|>import json
import os
from pathlib import Path
from pickle import DEFAULT_PROTOCOL, PicklingError
from typing import Any

from packaging import version

from huggingface_hub import constants, snapshot_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    get_fastai_version,
    get_fastcore_version,
    get_python_version,
)

from .utils import logging, validate_hf_hub_args


logger = logging.get_logger(__name__)


def _check_fastai_fastcore_versions(
    fastai_min_version: str = "2.4",
    fastcore_min_version: str = "1.3.27",
):
    """
    Checks that the installed fastai and fastcore versions are compatible for pickle serialization.

    Args:
        fastai_min_version (`str`, *optional*):
            The minimum fastai version supported.
        fastcore_min_version (`str`, *optional*):
            The minimum fastcore version supported.

    > [!TIP]
    > Raises the following error:
    >
    >     - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
    >       if the fastai or fastcore libraries are not available or are of an invalid version.
    """

    if (get_fastcore_version() or get_fastai_version()) == "N/A":
        raise ImportError(
            f"fastai>={fastai_min_version} and fastcore>={fastcore_min_version} are"
            f" required. Currently using fastai=={get_fastai_version()} and"
            f" fastcore=={get_fastcore_version()}."
        )

    current_fastai_version = version.Version(get_fastai_version())
    current_fastcore_version = version.Version(get_fastcore_version())

    if current_fastai_version < version.Version(fastai_min_version):
        raise ImportError(
            "`push_to_hub_fastai` and `from_pretrained_fastai` require a"
            f" fastai>={fastai_min_version} version, but you are using fastai version"
            f" {get_fastai_version()} which is incompatible. Upgrade with `pip install"
            " fastai==2.5.6`."
        )

    if current_fastcore_version < version.Version(fastcore_min_version):
        raise ImportError(
            "`push_to_hub_fastai` and `from_pretrained_fastai` require a"
            f" fastcore>={fastcore_min_version} version, but you are using fastcore"
            f" version {get_fastcore_version()} which is incompatible. Upgrade with"
            " `pip install fastcore==1.3.27`."
        )


def _check_fastai_fastcore_pyproject_versions(
    storage_folder: str,
    fastai_min_version: str = "2.4",
    fastcore_min_version: str = "1.3.27",
):
    """
    Checks that the `pyproject.toml` file in the directory `storage_folder` has fastai and fastcore versions
    that are compatible with `from_pretrained_fastai` and `push_to_hub_fastai`. If `pyproject.toml` does not exist
    or does not contain versions for fastai and fastcore, then it logs a warning.

    Args:
        storage_folder (`str`):
            Folder to look for the `pyproject.toml` file.
        fastai_min_version (`str`, *optional*):
            The minimum fastai version supported.
        fastcore_min_version (`str`, *optional*):
            The minimum fastcore version supported.

    > [!TIP]
    > Raises the following errors:
    >
    >     - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
    >       if the `toml` module is not installed.
    >     - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
    >       if the `pyproject.toml` indicates a lower than minimum supported version of fastai or fastcore.
    """

    try:
        import toml
    except ModuleNotFoundError:
        raise ImportError(
            "`push_to_hub_fastai` and `from_pretrained_fastai` require the toml module."
            " Install it with `pip install toml`."
        )

    # Checks that a `pyproject.toml`, with `build-system` and `requires` sections, exists in the repository. If so, get a list of required packages.
    if not os.path.isfile(f"{storage_folder}/pyproject.toml"):
        logger.warning(
            "There is no `pyproject.toml` in the repository that contains the fastai"
            " `Learner`. The `pyproject.toml` would allow us to verify that your fastai"
            " and fastcore versions are compatible with those of the model you want to"
            " load."
        )
        return
    pyproject_toml = toml.load(f"{storage_folder}/pyproject.toml")

    if "build-system" not in pyproject_toml.keys():
        logger.warning(
            "There is no `build-system` section in the pyproject.toml of the repository"
            " that contains the fastai `Learner`. The `build-system` would allow us to"
            " verify that your fastai and fastcore versions are compatible with those"
            " of the model you want to load."
        )
        return
    build_system_toml = pyproject_toml["build-system"]

    if "requires" not in build_system_toml.keys():
        logger.warning(
            "There is no `requires` section in the pyproject.toml of the repository"
            " that contains the fastai `Learner`. The `requires` would allow us to"
            " verify that your fastai and fastcore versions are compatible with those"
            " of the model you want to load."
        )
        return
    package_versions = build_system_toml["requires"]

    # Extracts contains fastai and fastcore versions from `pyproject.toml` if available.
    # If the package is specified but not the version (e.g. "fastai" instead of "fastai=2.4"), the default versions are the highest.
    fastai_packages = [pck for pck in package_versions if pck.startswith("fastai")]
    if len(fastai_packages) == 0:
        logger.warning("The repository does not have a fastai version specified in the `pyproject.toml`.")
    # fastai_version is an empty string if not specified
    else:
        fastai_version = str(fastai_packages[0]).partition("=")[2]
        if fastai_version != "" and version.Version(fastai_version) < version.Version(fastai_min_version):
            raise ImportError(
                "`from_pretrained_fastai` requires"
                f" fastai>={fastai_min_version} version but the model to load uses"
                f" {fastai_version} which is incompatible."
            )

    fastcore_packages = [pck for pck in package_versions if pck.startswith("fastcore")]
    if len(fastcore_packages) == 0:
        logger.warning("The repository does not have a fastcore version specified in the `pyproject.toml`.")
    # fastcore_version is an empty string if not specified
    else:
        fastcore_version = str(fastcore_packages[0]).partition("=")[2]
        if fastcore_version != "" and version.Version(fastcore_version) < version.Version(fastcore_min_version):
            raise ImportError(
                "`from_pretrained_fastai` requires"
                f" fastcore>={fastcore_min_version} version, but you are using fastcore"
                f" version {fastcore_version} which is incompatible."
            )


README_TEMPLATE = """---
tags:
- fastai
---

# Amazing!

🥳 Congratulations on hosting your fastai model on the Hugging Face Hub!

# Some next steps
1. Fill out this model card with more information (see the template below and the [documentation here](https://huggingface.co/docs/hub/model-repos))!

2. Create a demo in Gradio or Streamlit using 🤗 Spaces ([documentation here](https://huggingface.co/docs/hub/spaces)).

3. Join the fastai community on the [Fastai Discord](https://discord.com/invite/YKrxeNn)!

Greetings fellow fastlearner 🤝! Don't forget to delete this content from your model card.


---


# Model card

## Model description
More information needed

## Intended uses & limitations
More information needed

## Training and evaluation data
More information needed
"""

PYPROJECT_TEMPLATE = f"""[build-system]
requires = ["setuptools>=40.8.0", "wheel", "python={get_python_version()}", "fastai={get_fastai_version()}", "fastcore={get_fastcore_version()}"]
build-backend = "setuptools.build_meta:__legacy__"
"""


def _create_model_card(repo_dir: Path):
    """
    Creates a model card for the repository.

    Args:
        repo_dir (`Path`):
            Directory where model card is created.
    """
    readme_path = repo_dir / "README.md"

    if not readme_path.exists():
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(README_TEMPLATE)


def _create_model_pyproject(repo_dir: Path):
    """
    Creates a `pyproject.toml` for the repository.

    Args:
        repo_dir (`Path`):
            Directory where `pyproject.toml` is created.
    """
    pyproject_path = repo_dir / "pyproject.toml"

    if not pyproject_path.exists():
        with pyproject_path.open("w", encoding="utf-8") as f:
            f.write(PYPROJECT_TEMPLATE)


def _save_pretrained_fastai(
    learner,
    save_directory: str | Path,
    config: dict[str, Any] | None = None,
):
    """
    Saves a fastai learner to `save_directory` in pickle format using the default pickle protocol for the version of python used.

    Args:
        learner (`Learner`):
            The `fastai.Learner` you'd like to save.
        save_directory (`str` or `Path`):
            Specific directory in which you want to save the fastai learner.
        config (`dict`, *optional*):
            Configuration object. Will be uploaded as a .json file. Example: 'https://huggingface.co/espejelomar/fastai-pet-breeds-classification/blob/main/config.json'.

    > [!TIP]
    > Raises the following error:
    >
    >     - [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError)
    >       if the config file provided is not a dictionary.
    """
    _check_fastai_fastcore_versions()

    os.makedirs(save_directory, exist_ok=True)

    # if the user provides config then we update it with the fastai and fastcore versions in CONFIG_TEMPLATE.
    if config is not None:
        if not isinstance(config, dict):
            raise RuntimeError(f"Provided config should be a dict. Got: '{type(config)}'")
        path = os.path.join(save_directory, constants.CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(config, f)

    _create_model_card(Path(save_directory))
    _create_model_pyproject(Path(save_directory))

    # learner.export saves the model in `self.path`.
    learner.path = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    try:
        learner.export(
            fname="model.pkl",
            pickle_protocol=DEFAULT_PROTOCOL,
        )
    except PicklingError:
        raise PicklingError(
            "You are using a lambda function, i.e., an anonymous function. `pickle`"
            " cannot pickle function objects and requires that all functions have"
            " names. One possible solution is to name the function."
        )


@validate_hf_hub_args
def from_pretrained_fastai(
    repo_id: str,
    revision: str | None = None,
):
    """
    Load pretrained fastai model from the Hub or from a local directory.

    Args:
        repo_id (`str`):
            The location where the pickled fastai.Learner is. It can be either of the two:
                - Hosted on the Hugging Face Hub. E.g.: 'espejelomar/fatai-pet-breeds-classification' or 'distilgpt2'.
                  You can add a `revision` by appending `@` at the end of `repo_id`. E.g.: `dbmdz/bert-base-german-cased@main`.
                  Revision is the specific model version to use. Since we use a git-based system for storing models and other
                  artifacts on the Hugging Face Hub, it can be a branch name, a tag name, or a commit id.
                - Hosted locally. `repo_id` would be a directory containing the pickle and a pyproject.toml
                  indicating the fastai and fastcore versions used to build the `fastai.Learner`. E.g.: `./my_model_directory/`.
        revision (`str`, *optional*):
            Revision at which the repo's files are downloaded. See documentation of `snapshot_download`.

    Returns:
        The `fastai.Learner` model in the `repo_id` repo.
    """
    _check_fastai_fastcore_versions()

    # Load the `repo_id` repo.
    # `snapshot_download` returns the folder where the model was stored.
    # `cache_dir` will be the default '/root/.cache/huggingface/hub'
    if not os.path.isdir(repo_id):
        storage_folder = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            library_name="fastai",
            library_version=get_fastai_version(),
        )
    else:
        storage_folder = repo_id

    _check_fastai_fastcore_pyproject_versions(storage_folder)

    from fastai.learner import load_learner  # type: ignore

    return load_learner(os.path.join(storage_folder, "model.pkl"))


@validate_hf_hub_args
def push_to_hub_fastai(
    learner,
    *,
    repo_id: str,
    commit_message: str = "Push FastAI model using huggingface_hub.",
    private: bool | None = None,
    token: str | None = None,
    config: dict | None = None,
    branch: str | None = None,
    create_pr: bool | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    delete_patterns: list[str] | str | None = None,
    api_endpoint: str | None = None,
):
    """
    Upload learner checkpoint files to the Hub.

    Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
    `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
    details.

    Args:
        learner (`Learner`):
            The `fastai.Learner' you'd like to push to the Hub.
        repo_id (`str`):
            The repository id for your model in Hub in the format of "namespace/repo_name". The namespace can be your individual account or an organization to which you have write access (for example, 'stanfordnlp/stanza-de').
        commit_message (`str`, *optional*):
            Message to commit while pushing. Will default to :obj:`"add model"`.
        private (`bool`, *optional*):
            Whether or not the repository created should be private.
            If `None` (default), will default to been public except if the organization's default is private.
        token (`str`, *optional*):
            The Hugging Face account token to use as HTTP bearer authorization for remote files. If :obj:`None`, the token will be asked by a prompt.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        branch (`str`, *optional*):
            The git branch on which to push the model. This defaults to
            the default branch as specified in your repository, which
            defaults to `"main"`.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `branch` with that commit.
            Defaults to `False`.
        api_endpoint (`str`, *optional*):
            The API endpoint to use when pushing the model to the hub.
        allow_patterns (`list[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are pushed.
        ignore_patterns (`list[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not pushed.
        delete_patterns (`list[str]` or `str`, *optional*):
            If provided, remote files matching any of the patterns will be deleted from the repo.

    Returns:
        The url of the commit of your model in the given repository.

    > [!TIP]
    > Raises the following error:
    >
    >     - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
    >       if the user is not log on to the Hugging Face Hub.
    """
    _check_fastai_fastcore_versions()
    api = HfApi(endpoint=api_endpoint)
    repo_id = api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True).repo_id

    # Push the files to the repo in a single commit
    with SoftTemporaryDirectory() as tmp:
        saved_path = Path(tmp) / repo_id
        _save_pretrained_fastai(learner, saved_path, config=config)
        return api.upload_folder(
            repo_id=repo_id,
            token=token,
            folder_path=saved_path,
            commit_message=commit_message,
            revision=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
        )
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
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from huggingface_hub.utils import HfMount, HfUri, parse_datetime


class SpaceStage(str, Enum):
    """
    Enumeration of possible stage of a Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceStage.BUILDING == "BUILDING"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L61 (private url).
    """

    # Copied from moon-landing > server > repo_types > SpaceInfo.ts (private repo)
    NO_APP_FILE = "NO_APP_FILE"
    CONFIG_ERROR = "CONFIG_ERROR"
    BUILDING = "BUILDING"
    BUILD_ERROR = "BUILD_ERROR"
    RUNNING = "RUNNING"
    RUNNING_BUILDING = "RUNNING_BUILDING"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    DELETING = "DELETING"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    APP_STARTING = "APP_STARTING"
    RUNNING_APP_STARTING = "RUNNING_APP_STARTING"


INTERMEDIATE_SPACE_STAGES = (
    SpaceStage.BUILDING,
    SpaceStage.RUNNING_BUILDING,
    SpaceStage.APP_STARTING,
    SpaceStage.RUNNING_APP_STARTING,
)

TERMINAL_SPACE_STAGES = (
    SpaceStage.RUNNING,
    SpaceStage.BUILD_ERROR,
    SpaceStage.RUNTIME_ERROR,
    SpaceStage.CONFIG_ERROR,
    SpaceStage.NO_APP_FILE,
    SpaceStage.STOPPED,
    SpaceStage.PAUSED,
    SpaceStage.DELETING,
)


class SpaceHardware(str, Enum):
    """
    Enumeration of hardwares available to run your Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceHardware.CPU_BASIC == "cpu-basic"
    ```

    Taken from https://github.com/huggingface-internal/moon-landing/blob/main/server/repo_types/SpaceHardwareFlavor.ts (private url).
    """

    # CPU
    CPU_BASIC = "cpu-basic"
    CPU_UPGRADE = "cpu-upgrade"

    # ZeroGPU
    ZERO_A10G = "zero-a10g"

    # GPU
    T4_SMALL = "t4-small"
    T4_MEDIUM = "t4-medium"
    L4X1 = "l4x1"
    L4X4 = "l4x4"
    L40SX1 = "l40sx1"
    L40SX4 = "l40sx4"
    L40SX8 = "l40sx8"
    A10G_SMALL = "a10g-small"
    A10G_LARGE = "a10g-large"
    A10G_LARGEX2 = "a10g-largex2"
    A10G_LARGEX4 = "a10g-largex4"
    A100_LARGE = "a100-large"
    A100X4 = "a100x4"
    A100X8 = "a100x8"


class SpaceStorage(str, Enum):
    """
    Enumeration of persistent storage available for your Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceStorage.SMALL == "small"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceHardwareFlavor.ts#L24 (private url).
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class Volume:
    """
    Describes a volume to mount in a Space or Job container.

    Args:
        type (`str`):
            Type of volume: `"bucket"`, `"model"`, `"dataset"`, or `"space"`.
        source (`str`):
            Source identifier, e.g. `"username/my-bucket"` or `"username/my-model"`.
        mount_path (`str`):
            Mount path inside the container, e.g. `"/data"`. Must start with `/`.
        revision (`str` or `None`):
            Git revision (only for repos, defaults to `"main"`).
        read_only (`bool` or `None`):
            Read-only mount. Forced `True` for repos, defaults to `False` for buckets.
        path (`str` or `None`):
            Subfolder prefix inside the bucket/repo to mount, e.g. `"path/to/dir"`.
    """

    type: Literal["bucket", "model", "dataset", "space"]
    source: str
    mount_path: str
    revision: str | None = None
    read_only: bool | None = None
    path: str | None = None

    def __init__(self, **kwargs) -> None:
        self.type = kwargs.get("type", "model")
        self.source = kwargs["source"]
        mount_path = kwargs.get("mountPath")
        self.mount_path = mount_path if mount_path is not None else kwargs["mount_path"]
        self.revision = kwargs.get("revision")
        read_only = kwargs.get("readOnly")
        self.read_only = read_only if read_only is not None else kwargs.get("read_only")
        self.path = kwargs.get("path")

    def to_dict(self) -> dict:
        """Serialize to the JSON payload expected by the Hub API."""
        data: dict = {
            "type": self.type,
            "source": self.source,
            "mountPath": self.mount_path,
        }
        if self.revision is not None:
            data["revision"] = self.revision
        if self.read_only is not None:
            data["readOnly"] = self.read_only
        if self.path is not None:
            data["path"] = self.path
        return data

    def to_uri(self) -> str:
        """Return the volume as an HF mount URI in the format expected by the CLI."""
        return HfMount(
            source=HfUri(type=self.type, id=self.source, revision=self.revision, path_in_repo=self.path or ""),
            mount_path=self.mount_path,
            read_only=self.read_only,
        ).to_uri()


@dataclass
class SpaceHotReloading:
    status: Literal["created", "canceled"]
    replica_statuses: list[tuple[str, str | None]]  # See _hot_reloading_types.ApiCreateReloadResponse.res.status
    raw: dict

    def __init__(self, data: dict) -> None:
        self.status = data["status"]
        self.replica_statuses = data["replicaStatuses"]
        self.raw = data


@dataclass
class SpaceRuntime:
    """
    Contains information about the current runtime of a Space.

    Args:
        stage (`str`):
            Current stage of the space. Example: RUNNING.
        hardware (`str` or `None`):
            Current hardware of the space. Example: "cpu-basic". Can be `None` if Space
            is `BUILDING` for the first time.
        requested_hardware (`str` or `None`):
            Requested hardware. Can be different from `hardware` especially if the request
            has just been made. Example: "t4-medium". Can be `None` if no hardware has
            been requested yet.
        sleep_time (`int` or `None`):
            Number of seconds the Space will be kept alive after the last request. By default (if value is `None`), the
            Space will never go to sleep if it's running on an upgraded hardware, while it will go to sleep after 48
            hours on a free 'cpu-basic' hardware. For more details, see https://huggingface.co/docs/hub/spaces-gpus#sleep-time.
        volumes (`list[Volume]` or `None`):
            List of volumes mounted in the Space. Each volume is a [`Volume`] object describing its type, source,
            mount path, and optional settings. `None` if no volumes are attached.
        raw (`dict`):
            Raw response from the server. Contains more information about the Space
            runtime like number of replicas, number of cpu, memory size,...
    """

    stage: SpaceStage
    hardware: SpaceHardware | None
    requested_hardware: SpaceHardware | None
    sleep_time: int | None
    storage: SpaceStorage | None
    dev_mode: bool
    hot_reloading: SpaceHotReloading | None
    volumes: list[Volume] | None
    raw: dict

    def __init__(self, data: dict) -> None:
        self.stage = data["stage"]
        hardware_dict = data.get("hardware") or {}
        self.hardware = hardware_dict.get("current")
        self.requested_hardware = hardware_dict.get("requested")
        self.sleep_time = data.get("gcTimeout")
        self.storage = data.get("storage")
        self.dev_mode = data.get("devMode", False)
        self.hot_reloading = SpaceHotReloading(raw_hr) if (raw_hr := data.get("hotReloading")) is not None else None
        raw_volumes = data.get("volumes")
        self.volumes = [Volume(**v) for v in raw_volumes] if raw_volumes is not None else None
        self.raw = data


@dataclass
class SpaceSecret:
    """
    Contains information about a secret of a Space.

    Secret values are write-only and cannot be read back. Only the key, description,
    and last update time are returned by the API.

    Args:
        key (`str`):
            Secret key. Example: `"GITHUB_API_KEY"`
        description (`str` or None):
            Description of the secret. Example: `"Github API key to access the Github API"`.
        updated_at (`datetime` or None):
            datetime of the last update of the secret (if the secret has been updated at least once).
    """

    key: str
    description: str | None
    updated_at: datetime | None

    def __init__(self, key: str, values: dict) -> None:
        self.key = key
        self.description = values.get("description")
        updated_at = values.get("updatedAt")
        self.updated_at = parse_datetime(updated_at) if updated_at is not None else None


@dataclass
class SpaceVariable:
    """
    Contains information about the current variables of a Space.

    Args:
        key (`str`):
            Variable key. Example: `"MODEL_REPO_ID"`
        value (`str`):
            Variable value. Example: `"the_model_repo_id"`.
        description (`str` or None):
            Description of the variable. Example: `"Model Repo ID of the implemented model"`.
        updatedAt (`datetime` or None):
            datetime of the last update of the variable (if the variable has been updated at least once).
    """

    key: str
    value: str
    description: str | None
    updated_at: datetime | None

    def __init__(self, key: str, values: dict) -> None:
        self.key = key
        self.value = values["value"]
        self.description = values.get("description")
        updated_at = values.get("updatedAt")
        self.updated_at = parse_datetime(updated_at) if updated_at is not None else None


@dataclass
class SpaceSearchResult:
    """A single result from the Spaces semantic search API.

    Returned by [`HfApi.search_spaces`].

    Attributes:
        id (`str`):
            ID of the Space (e.g. `"username/repo-name"`).
        author (`str`):
            Author of the Space.
        title (`str`):
            Display title of the Space.
        emoji (`str` or `None`):
            Emoji icon of the Space.
        sdk (`str` or `None`):
            SDK used by the Space (e.g. `"gradio"`, `"docker"`, `"static"`).
        likes (`int`):
            Number of likes.
        private (`bool`):
            Whether the Space is private.
        tags (`list[str]` or `None`):
            List of tags.
        runtime ([`SpaceRuntime`] or `None`):
            Runtime information (stage, hardware, etc.).
        ai_short_description (`str` or `None`):
            AI-generated short description.
        ai_category (`str` or `None`):
            AI-generated category (e.g. `"Image Generation"`).
        semantic_relevancy_score (`float` or `None`):
            Semantic relevancy score (0-1) relative to the search query.
        trending_score (`int` or `None`):
            Trending score.
    """

    id: str
    author: str
    title: str
    emoji: str | None
    sdk: str | None
    likes: int
    private: bool
    tags: list[str] | None
    runtime: SpaceRuntime | None
    ai_short_description: str | None
    ai_category: str | None
    semantic_relevancy_score: float | None
    trending_score: int | None

    def __init__(self, data: dict) -> None:
        runtime = data.get("runtime")
        self.id = data["id"]
        self.author = data.get("author", "")
        self.title = data.get("title", "")
        self.emoji = data.get("emoji")
        self.sdk = data.get("sdk")
        self.likes = data.get("likes", 0)
        self.private = data.get("private", False)
        self.tags = data.get("tags")
        self.runtime = SpaceRuntime(runtime) if runtime else None
        self.ai_short_description = data.get("ai_short_description")
        self.ai_category = data.get("ai_category")
        self.semantic_relevancy_score = data.get("semanticRelevancyScore")
        self.trending_score = data.get("trendingScore")


@dataclass
class SpaceTemplate:
    """
    Contains information about a Space template available on the Hub.

    Returned by [`HfApi.list_space_templates`]. The `repo_id` can be passed as `space_template`
    to [`HfApi.create_repo`] to seed a new Space from that template.

    Args:
        name (`str`):
            Human-friendly name of the template (e.g. `"JupyterLab"`, `"chatbot"`).
        repo_id (`str`):
            Repo id of the template Space (e.g. `"SpacesExamples/jupyterlab"`).
        sdk (`str`):
            SDK the template is built with (e.g. `"gradio"`, `"docker"`, `"static"`).
        preferred_private (`bool`):
            Whether Spaces created from this template are recommended to be private.
    """

    name: str
    repo_id: str
    sdk: str
    preferred_private: bool

    def __init__(self, data: dict) -> None:
        self.name = data["name"]
        self.repo_id = data["repoId"]
        self.sdk = data["sdk"]
        self.preferred_private = data["preferredPrivate"]


@dataclass
class ZeroGpuQuota:
    """
    Contains information about the ZeroGPU quota of the authenticated user.

    The ZeroGPU quota is consumed each time you use a ZeroGPU Space (from the UI or through its API) and is reset
    periodically. See https://huggingface.co/docs/hub/spaces-zerogpu for more details.

    Returned by [`HfApi.get_zero_gpu_quota`].

    Args:
        base (`float`):
            Total quota, in GPU-seconds.
        remaining (`float`):
            Remaining quota, in GPU-seconds.
        resets_at (`datetime` or `None`):
            Date at which the quota will be reset. `None` if the quota has not been used yet.
        overquota_used (`float` or `None`):
            Overquota (paid) GPU-seconds consumed once the base quota is exhausted. PRO, Team and Enterprise users
            can purchase credits to go beyond the base quota at https://huggingface.co/settings/billing?add-credits=true.
    """

    base: float
    remaining: float
    resets_at: datetime | None
    overquota_used: float | None

    def __init__(self, data: dict) -> None:
        self.base = data["base"]
        self.remaining = data["current"]
        self.resets_at = parse_datetime(resets_at) if (resets_at := data.get("resetsAt")) is not None else None
        self.overquota_used = data.get("overquotaUsed")
<|endoftext|>import atexit
import logging
import os
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

from ._commit_api import CommitOperationAdd
from .hf_api import DEFAULT_IGNORE_PATTERNS, CommitInfo, HfApi
from .utils import filter_repo_objects


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FileToUpload:
    """Temporary dataclass to store info about files to upload. Not meant to be used directly."""

    local_path: Path
    path_in_repo: str
    size_limit: int
    last_modified: float


class CommitScheduler:
    """
    Scheduler to upload a local folder to the Hub at regular intervals (e.g. push to hub every 5 minutes).

    The recommended way to use the scheduler is to use it as a context manager. This ensures that the scheduler is
    properly stopped and the last commit is triggered when the script ends. The scheduler can also be stopped manually
    with the `stop` method. Checkout the [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads)
    to learn more about how to use it.

    Args:
        repo_id (`str`):
            The id of the repo to commit to.
        folder_path (`str` or `Path`):
            Path to the local folder to upload regularly.
        every (`int` or `float`, *optional*):
            The number of minutes between each commit. Defaults to 5 minutes.
        path_in_repo (`str`, *optional*):
            Relative path of the directory in the repo, for example: `"checkpoints/"`. Defaults to the root folder
            of the repository.
        repo_type (`str`, *optional*):
            The type of the repo to commit to. Defaults to `model`.
        revision (`str`, *optional*):
            The revision of the repo to commit to. Defaults to `main`.
        private (`bool`, *optional*):
            Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
        token (`str`, *optional*):
            The token to use to commit to the repo. Defaults to the token saved on the machine.
        allow_patterns (`list[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are uploaded.
        ignore_patterns (`list[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not uploaded.
        squash_history (`bool`, *optional*):
            Whether to squash the history of the repo after each commit. Defaults to `False`. Squashing commits is
            useful to avoid degraded performances on the repo when it grows too large.
        hf_api (`HfApi`, *optional*):
            The [`HfApi`] client to use to commit to the Hub. Can be set with custom settings (user agent, token,...).

    Example:
    ```py
    >>> from pathlib import Path
    >>> from huggingface_hub import CommitScheduler

    # Scheduler uploads every 10 minutes
    >>> csv_path = Path("watched_folder/data.csv")
    >>> CommitScheduler(repo_id="test_scheduler", repo_type="dataset", folder_path=csv_path.parent, every=10)

    >>> with csv_path.open("a") as f:
    ...     f.write("first line")

    # Some time later (...)
    >>> with csv_path.open("a") as f:
    ...     f.write("second line")
    ```

    Example using a context manager:
    ```py
    >>> from pathlib import Path
    >>> from huggingface_hub import CommitScheduler

    >>> with CommitScheduler(repo_id="test_scheduler", repo_type="dataset", folder_path="watched_folder", every=10) as scheduler:
    ...     csv_path = Path("watched_folder/data.csv")
    ...     with csv_path.open("a") as f:
    ...         f.write("first line")
    ...     (...)
    ...     with csv_path.open("a") as f:
    ...         f.write("second line")

    # Scheduler is now stopped and last commit have been triggered
    ```
    """

    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        every: int | float = 5,
        path_in_repo: str | None = None,
        repo_type: str | None = None,
        revision: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        squash_history: bool = False,
        hf_api: Optional["HfApi"] = None,
    ) -> None:
        self.api = hf_api or HfApi(token=token)

        # Folder
        self.folder_path = Path(folder_path).expanduser().resolve()
        self.path_in_repo = path_in_repo or ""
        self.allow_patterns = allow_patterns

        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        self.ignore_patterns = ignore_patterns + DEFAULT_IGNORE_PATTERNS

        if self.folder_path.is_file():
            raise ValueError(f"'folder_path' must be a directory, not a file: '{self.folder_path}'.")
        self.folder_path.mkdir(parents=True, exist_ok=True)

        # Repository
        repo_url = self.api.create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
        self.repo_id = repo_url.repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = token

        # Keep track of already uploaded files
        self.last_uploaded: dict[Path, float] = {}  # key is local path, value is timestamp

        # Scheduler
        if not every > 0:
            raise ValueError(f"'every' must be a positive integer, not '{every}'.")
        self.lock = Lock()
        self.every = every
        self.squash_history = squash_history
        self.__stopped = False

        logger.info(f"Scheduled job to push '{self.folder_path}' to '{self.repo_id}' every {self.every} minutes.")
        self._scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        atexit.register(self._push_to_hub)

    def stop(self) -> None:
        """Stop the scheduler.

        A stopped scheduler cannot be restarted. Mostly for tests purposes.
        """
        self.__stopped = True

    def __enter__(self) -> "CommitScheduler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Upload last changes before exiting
        self.trigger().result()
        self.stop()
        return

    def _run_scheduler(self) -> None:
        """Dumb thread waiting between each scheduled push to Hub."""
        while True:
            self.last_future = self.trigger()
            time.sleep(self.every * 60)
            if self.__stopped:
                break

    def trigger(self) -> Future:
        """Trigger a `push_to_hub` and return a future.

        This method is automatically called every `every` minutes. You can also call it manually to trigger a commit
        immediately, without waiting for the next scheduled commit.
        """
        return self.api.run_as_future(self._push_to_hub)

    def _push_to_hub(self) -> CommitInfo | None:
        if self.__stopped:  # If stopped, already scheduled commits are ignored
            return None

        logger.info("(Background) scheduled commit triggered.")
        try:
            value = self.push_to_hub()
            if self.squash_history:
                logger.info("(Background) squashing repo history.")
                self.api.super_squash_history(repo_id=self.repo_id, repo_type=self.repo_type, branch=self.revision)
            return value
        except Exception as e:
            logger.error(f"Error while pushing to Hub: {e}")  # Depending on the setup, error might be silenced
            raise

    def push_to_hub(self) -> CommitInfo | None:
        """
        Push folder to the Hub and return the commit info.

        > [!WARNING]
        > This method is not meant to be called directly. It is run in the background by the scheduler, respecting a
        > queue mechanism to avoid concurrent commits. Making a direct call to the method might lead to concurrency
        > issues.

        The default behavior of `push_to_hub` is to assume an append-only folder. It lists all files in the folder and
        uploads only changed files. If no changes are found, the method returns without committing anything. If you want
        to change this behavior, you can inherit from [`CommitScheduler`] and override this method. This can be useful
        for example to compress data together in a single file before committing. For more details and examples, check
        out our [integration guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads).
        """
        # Check files to upload (with lock)
        with self.lock:
            logger.debug("Listing files to upload for scheduled commit.")

            # List files from folder (taken from `_prepare_upload_folder_additions`)
            relpath_to_abspath = {
                path.relative_to(self.folder_path).as_posix(): path
                for path in sorted(self.folder_path.glob("**/*"))  # sorted to be deterministic
                if path.is_file()
            }
            prefix = f"{self.path_in_repo.strip('/')}/" if self.path_in_repo else ""

            # Filter with pattern + filter out unchanged files + retrieve current file size
            files_to_upload: list[_FileToUpload] = []
            for relpath in filter_repo_objects(
                relpath_to_abspath.keys(), allow_patterns=self.allow_patterns, ignore_patterns=self.ignore_patterns
            ):
                local_path = relpath_to_abspath[relpath]
                stat = local_path.stat()
                if self.last_uploaded.get(local_path) is None or self.last_uploaded[local_path] != stat.st_mtime:
                    files_to_upload.append(
                        _FileToUpload(
                            local_path=local_path,
                            path_in_repo=prefix + relpath,
                            size_limit=stat.st_size,
                            last_modified=stat.st_mtime,
                        )
                    )

        # Return if nothing to upload
        if len(files_to_upload) == 0:
            logger.debug("Dropping schedule commit: no changed file to upload.")
            return None

        # Convert `_FileToUpload` as `CommitOperationAdd` (=> compute file shas + limit to file size)
        logger.debug("Removing unchanged files since previous scheduled commit.")
        add_operations = [
            CommitOperationAdd(
                # Cap the file to its current size, even if the user append data to it while a scheduled commit is happening
                path_or_fileobj=PartialFileIO(file_to_upload.local_path, size_limit=file_to_upload.size_limit),
                path_in_repo=file_to_upload.path_in_repo,
            )
            for file_to_upload in files_to_upload
        ]

        # Upload files (append mode expected - no need for lock)
        logger.debug("Uploading files for scheduled commit.")
        commit_info = self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=add_operations,
            commit_message="Scheduled Commit",
            revision=self.revision,
        )

        # Successful commit: keep track of the latest "last_modified" for each file
        for file in files_to_upload:
            self.last_uploaded[file.local_path] = file.last_modified
        return commit_info


class PartialFileIO(BytesIO):
    """A file-like object that reads only the first part of a file.

    Useful to upload a file to the Hub when the user might still be appending data to it. Only the first part of the
