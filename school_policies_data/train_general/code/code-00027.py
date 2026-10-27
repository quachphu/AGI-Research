            raise click.BadParameter("Stdin upload requires a full destination path including filename.")
        data = sys.stdin.buffer.read()
        _upload_single(api, uri, data, uri.path_in_repo)
        out.result("Uploaded", src="stdin", dst=uri.to_uri())
        return

    if os.path.isdir(src):
        raise click.BadParameter(
            "Source must be a file, not a directory. Use `hf upload` or `hf buckets sync` for directories."
        )
    if not os.path.isfile(src):
        raise click.BadParameter(f"Source file not found: {src}")

    prefix = uri.path_in_repo
    if prefix == "":
        remote_path = os.path.basename(src)
    elif dst.endswith("/"):
        remote_path = prefix + "/" + os.path.basename(src)
    else:
        remote_path = prefix

    _upload_single(api, uri, src, remote_path)
    out.result("Uploaded", src=src, dst=replace(uri, path_in_repo=remote_path).to_uri())


def _upload_single(api: HfApi, uri: HfUri, source: str | bytes, remote_path: str) -> None:
    """Upload a single file or bytes (to a repo or bucket)."""
    if uri.is_bucket:
        api.batch_bucket_files(uri.id, add=[(source, remote_path)])
    else:
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=remote_path,
            repo_id=uri.id,
            repo_type=uri.type,
            revision=uri.revision,
        )
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
"""Contains commands to interact with jobs on the Hugging Face Hub."""

import itertools
import multiprocessing
import multiprocessing.pool
import shlex
import shutil
import time
from collections.abc import Callable, Collection, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Annotated, Any, TypeVar
from urllib.parse import urlsplit

from huggingface_hub import HfApi, JobHardware, JobInfo, JobNetwork, JobStage, Volume, constants
from huggingface_hub._jobs_api import (
    DEFAULT_UV_IMAGE,
    TERMINAL_JOB_STAGES,
    _default_job_name_from_image,
    _default_job_name_from_script,
)
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import logging
from huggingface_hub.utils._cache_manager import _format_size
from huggingface_hub.utils._hf_uris import _split_mount
from huggingface_hub.utils._parsing import format_duration, parse_duration

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    FormatOpt,
    JsonOpt,
    QuietOpt,
    SecretsFileOpt,
    SecretsOpt,
    SoftChoice,
    SshDryRunOpt,
    SshIdentityFileOpt,
    TokenOpt,
    _get_extended_environ,
    exec_ssh,
    get_hf_api,
    parse_env_map,
    parse_volumes,
    set_output_format,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import OutputFormat, _dataclass_to_dict, out
from ._uv_script_header import TABLE_NAME, UvScriptHeader, load_uv_script


logger = logging.get_logger(__name__)


def _parse_namespace_from_job_id(job_id: str, namespace: str | None) -> tuple[str, str | None]:
    """Extract namespace from job_id if provided in 'namespace/job_id' format.

    Allows users to pass job IDs copied from the Hub UI (e.g. 'username/job_id')
    instead of only bare job IDs. If the namespace is also provided explicitly via
    --namespace and conflicts, a CLIError is raised.
    """
    if not job_id:
        raise CLIError("Job ID cannot be empty.")

    if job_id.count("/") > 1:
        raise CLIError(f"Job ID must be in the form 'job_id' or 'namespace/job_id': '{job_id}'.")

    if "/" not in job_id:
        return job_id, namespace

    extracted_namespace, parsed_job_id = job_id.split("/", 1)
    if not extracted_namespace or not parsed_job_id:
        raise CLIError(f"Job ID must be in the form 'job_id' or 'namespace/job_id': '{job_id}'.")

    if namespace is not None and namespace != extracted_namespace:
        raise CLIError(
            f"Conflicting namespace: got --namespace='{namespace}' but job ID implies namespace='{extracted_namespace}'"
        )

    return parsed_job_id, extracted_namespace


def _parse_and_sync_job_volumes(
    volumes: list[str] | None, *, api: HfApi, namespace: str | None
) -> list[Volume] | None:
    """Parse `-v` specs for Jobs commands.

    Same as [`parse_volumes`] but the source side can also be a local directory: it is synced to a
    bucket via [`HfApi.sync_job_volume`] and the resulting bucket subfolder is mounted (read-only
    unless ':rw' is specified).
    """
    if not volumes:
        return None

    result: list[Volume] = []
    for raw_spec in volumes:
        if raw_spec.startswith(constants.HF_PROTOCOL):
            result.extend(parse_volumes([raw_spec]) or [])
            continue

        # Not a 'hf://' URI: treat the source as a local directory.
        source, mount_path, read_only = _split_mount(raw_spec, raw=raw_spec)
        if mount_path is None:
            raise CLIError(
                f"Missing mount path in volume spec '{raw_spec}'. Expected 'LOCAL_DIR:/MOUNT_PATH[:ro|:rw]' (e.g. './data:/data')."
            )
        if not Path(source).expanduser().is_dir():
            raise CLIError(
                f"Volume source '{source}' is not an existing local directory. "
                "To mount a repo or bucket instead, use the 'hf://' syntax (e.g. 'hf://buckets/my-org/my-bucket:/data')."
            )
        volume = api.sync_job_volume(
            source,
            mount_path,
            read_only=read_only if read_only is not None else True,
            namespace=namespace,
        )
        if volume.read_only is False:
            out.hint(
                f"Volume '{mount_path}' is mounted read-write. Once the job is over, pull back its data with:\n"
                f"  hf buckets sync hf://buckets/{volume.source}/{volume.path} {source}"
            )
        result.append(volume)
    return result


@dataclass
class _UvJobConfig:
    """Resolved launch configuration of a UV Job: CLI flags merged with the script's `[tool.hf-jobs]` table."""

    script: str
    """What is passed to the Jobs API. Same as the CLI argument, except for a URL script (downloaded locally)."""

    script_args: list[str] = field(default_factory=list)
    image: str | None = None
    flavor: str | None = None
    python: str | None = None
    timeout: str | None = None
    namespace: str | None = None
    network_group: str | None = None
    env: dict[str, str | None] = field(default_factory=dict)
    secrets: dict[str, str | None] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    volume_specs: list[str] = field(default_factory=list)
    volumes: list[Volume] | None = None
    network_aliases: list[str] = field(default_factory=list)

    from_script: set[str] = field(default_factory=set)
    """Keys whose value comes from the script's `[tool.hf-jobs]` table, for display purposes."""


@contextmanager
def _resolve_uv_job_config(
    *,
    api: HfApi,
    script: str,
    script_args: list[str] | None,
    dependencies: list[str] | None,
    image: str | None,
    flavor: str | None,
    python: str | None,
    env: list[str] | None,
    env_file: str | None,
    secrets: list[str] | None,
    secrets_file: str | None,
    timeout: str | None,
    attempts: int | None,
    name: str | None,
    label: list[str] | None,
    volume: list[str] | None,
    namespace: str | None,
    network_group: str | None,
    network_aliases: list[str] | None,
    dry_run: bool,
) -> Generator[_UvJobConfig, None, None]:
    """Merge the CLI flags with the `[tool.hf-jobs]` table the script carries, if any.

    An explicit CLI flag always wins over the script; `env`, `labels`, `secrets` and `volumes` are merged
    entry by entry (by key, by name and by mount path respectively), so a script value and a CLI value
    only ever conflict when they target the same entry. `network_aliases` is the exception: the aliases
    a Job claims form a set, so `--network-alias` replaces the script's list rather than adding to it.

    A URL script is downloaded to a temporary file that lives until the `with` block exits: submit the
    Job inside the block.
    """
    with load_uv_script(script) as source:
        header = source.header or UvScriptHeader()
        from_script: set[str] = set()

        def pick(key: str, cli_value: str | None) -> str | None:
            """Keep the CLI value, and fall back to the script's."""
            if cli_value is not None:
                return cli_value
            if (script_value := getattr(header, key)) is not None:
                from_script.add(key)
                return script_value
            return None

        image = pick("image", image)
        flavor = pick("flavor", flavor)
        python = pick("python", python)
        timeout = pick("timeout", timeout)
        namespace = pick("namespace", namespace)
        network_group = pick("network_group", network_group)

        if not network_aliases and header.network_aliases:
            from_script.add("network_aliases")
            network_aliases = header.network_aliases

        env_map = parse_env_map(env, env_file)
        script_env = {key: value for key, value in header.env.items() if key not in env_map}
        from_script.update(f"env.{key}" for key in script_env)
        env_map = {**script_env, **env_map}

        secrets_map = _parse_secrets_map(secrets, secrets_file, dry_run=dry_run)
        script_secrets = _resolve_script_secrets(header.secrets, secrets_map, dry_run=dry_run)
        from_script.update(f"secrets.{key}" for key in script_secrets)
        secrets_map = {**script_secrets, **secrets_map}

        labels_map = _parse_labels_map(label, name=name) or {}
        # `name = ...` in the script's table is shorthand for the `name` label.
        script_labels = {**header.labels, **({"name": header.name} if header.name is not None else {})}
        script_labels = {key: value for key, value in script_labels.items() if key not in labels_map}
        from_script.update(f"labels.{key}" for key in script_labels)
        labels_map = {**script_labels, **labels_map}

        volume_specs, script_volume_specs = _merge_volume_specs(volume or [], header.volumes)
        from_script.update(f"volumes.{spec}" for spec in script_volume_specs)

        config = _UvJobConfig(
            script=source.script,
            script_args=script_args or [],
            image=image,
            flavor=flavor,
            python=python,
            timeout=timeout,
            namespace=namespace,
            network_group=network_group,
            env=env_map,
            secrets=secrets_map,
            labels=labels_map,
            volume_specs=volume_specs,
            # A dry run must not have side effects: local directories are only synced to a bucket for real runs.
            volumes=None if dry_run else _parse_and_sync_job_volumes(volume_specs, api=api, namespace=namespace),
            network_aliases=network_aliases or [],
            from_script=from_script,
        )
        if "name" not in config.labels:
            # `script` (not `config.script`) so that a URL keeps naming the Job after the URL, not after
            # the temporary file it was downloaded to.
            config.labels["name"] = _default_job_name_from_script(
                script,
                config.script_args,
                config_parts=_name_hash_parts(
                    flavor=config.flavor,
                    timeout=config.timeout,
                    namespace=config.namespace,
                    env=config.env,
                    secrets=config.secrets,
                    volume_specs=config.volume_specs,
                    network_group=config.network_group,
                    network_aliases=config.network_aliases,
                    extra=[
                        config.image or DEFAULT_UV_IMAGE,
                        config.python or "",
                        *(dependencies or []),
                        *([str(attempts)] if attempts is not None else []),
                    ],
                ),
            )
        yield config


def _parse_secrets_map(
    secrets: list[str] | None, secrets_file: str | None, *, dry_run: bool = False
) -> dict[str, str | None]:
    """Parse `--secrets`/`--secrets-file`, failing if a bare `--secrets NAME` is not set locally.

    Same behavior as for the secrets requested by a script (see `_resolve_script_secrets`): a missing secret
    is an error rather than silently dropped, except in `--dry-run` where it is displayed as `<not set>`.
    """
    secrets_map = parse_env_map(secrets, secrets_file)
    missing = [name for value in secrets or [] if "=" not in value and (name := value.strip()) not in secrets_map]
    if missing and not dry_run:
        raise CLIError(
            f"The following secret(s) are not set in your environment: {', '.join(missing)}."
            f" Export them locally (e.g. `export {missing[0]}=...`) or use `--secrets-file` otherwise."
        )
    return {**secrets_map, **dict.fromkeys(missing)}


def _resolve_script_secrets(
    names: list[str], cli_secrets: dict[str, str | None], *, dry_run: bool = False
) -> dict[str, str | None]:
    """Resolve the secrets requested by a script from the caller's environment.

    A script only lists secret *names*: values always come from whoever runs it (`HF_TOKEN` also
    resolves from `hf auth login`). A requested secret that is not set locally is an error rather than
    an empty value: forgetting to export a secret before launching is easy, and debugging the Job that
    results from it is not.

    In `--dry-run` a missing secret is not an error but a `None` value, displayed as `<not set>` in the
    summary: nothing is submitted, so the point is to see the full configuration of a script - including
    one whose secrets are not provisioned locally (yet).
    """
    if not names:
        return {}
    environ = _get_extended_environ()
    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for name in names:
        if name in cli_secrets:  # an explicit `--secrets NAME=...` wins
            continue
        if name in environ:
            resolved[name] = environ[name]
        else:
            missing.append(name)
    if missing and not dry_run:
        raise CLIError(
            f"The script requires the following secret(s), which are not set in your environment: {', '.join(missing)}."
            f" Export them locally (e.g. `export {missing[0]}=...`) or use `--secrets-file` otherwise."
        )
    if resolved:
        out.warning(
            f"The script's [{TABLE_NAME}] table requests {', '.join(resolved)}: the value(s) from your local"
            f" environment {'would be' if dry_run else 'will be'} sent to the Job."
        )
    return {**resolved, **dict.fromkeys(missing)}


def _merge_volume_specs(cli_specs: list[str], script_specs: list[str]) -> tuple[list[str], list[str]]:
    """Merge `-v` specs with the ones from the script. Returns `(all specs, specs from the script)`.

    A CLI mount overrides the script's mount at the same path.
    """
    cli_mount_paths = {_split_mount(spec, raw=spec)[1] for spec in cli_specs}
    kept = [spec for spec in script_specs if _split_mount(spec, raw=spec)[1] not in cli_mount_paths]
    return kept + cli_specs, kept


def _name_hash_parts(
    *,
    flavor: str | None,
    timeout: str | None,
    namespace: str | None,
    env: dict[str, str | None],
    secrets: dict[str, str | None],
    volume_specs: list[str],
    network_group: str | None = None,
    network_aliases: list[str] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """The resolved launch values hashed into a Job's default name, on top of its image/script.

    Same inputs for `hf jobs run` and `hf jobs uv run`, so that both name Jobs after what they actually
    submit (`extra` carries what is specific to UV runs). Env *values* are part of the hash since they
    change what the Job does; secrets only contribute their names, never their values.
    """
    return [
        flavor or JobHardware.CPU_BASIC.value,
        timeout or "",
        namespace or "",
        network_group or "",
        *(extra or []),
        *(f"{key}={value}" for key, value in sorted(env.items())),
        *sorted(secrets),
        *volume_specs,
        *(network_aliases or []),
    ]


_FROM_SCRIPT = " (from script)"


def _print_job_summary(values: dict[str, Any], *, from_script: Collection[str] = (), dry_run: bool = False) -> None:
    """Echo the launch configuration of a Job before submitting it.

    Printed on every run, so that what is sent to the Jobs API is always visible - in particular the
    values injected by a script's `[tool.hf-jobs]` table, which are marked as such. Goes to stderr for a
    real run, and to stdout for `--dry-run`, where it is the output of the command.
    """
    rows = [
        (key, _format_job_value(key, value, from_script))
        for key, value in values.items()
        if value not in (None, "", [], {}, False)
    ]
    width = max(len(key) for key, _ in rows)
    lines = ["Job configuration:"]
    for key, value in rows:
        value += _FROM_SCRIPT if key in from_script else ""
        # Only the first line of a multi-line value is prefixed with the key.
        lines += [
            f"  {(key if index == 0 else '').ljust(width)}  {line}" for index, line in enumerate(value.split("\n"))
        ]
    summary = "\n".join(lines)
    if dry_run:
        out.text(f"{summary}\n(dry run) Job not submitted.")
    else:
        out.log(summary)


def _format_job_value(key: str, value: Any, from_script: Collection[str]) -> str:
    """Format config entries, redacting secrets and marking values inherited from the script."""
    match key:
        case "env":
            # Keep long prompts or JSON values readable in the summary.
            entries = {}
            for name, item in value.items():
                text = item or ""
                entries[name] = f"{name}={text[:59] + '…' if len(text) > 60 else text}"
        case "secrets":
            entries = {name: f"{name}={'***' if item is not None else '<not set>'}" for name, item in value.items()}
        case "labels":
            entries = {name: f"{name}={item}" for name, item in value.items()}
        case "volumes":
            entries = {spec: spec for spec in value}
        case _:
            return str(value)
    return "\n".join(text + (_FROM_SCRIPT if f"{key}.{name}" in from_script else "") for name, text in entries.items())


STATS_UPDATE_MIN_INTERVAL = 0.1  # we set a limit here since there is one update per second per job

# Common job-related options
ImageArg = Annotated[
    str,
    Argument(
        help="The Docker image to use.",
    ),
]

ImageOpt = Annotated[
    str | None,
    Option(
        help="Use a custom Docker image with `uv` installed.",
    ),
]

FlavorOpt = Annotated[
    str | None,
    Option(
        help="Flavor for the hardware. Run 'hf jobs hardware' to list available flavors. Defaults to `cpu-basic`.",
        click_type=SoftChoice(JobHardware),
    ),
]

LabelsOpt = Annotated[
    list[str] | None,
    Option(
        "-l",
        "--label",
        help="Set labels. E.g. --label KEY=VALUE or --label LABEL",
    ),
]

NameOpt = Annotated[
    str | None,
    Option(
        "--name",
        help="Name the Job. Stored as the `name` label. Names do not have to be unique. Defaults to the image or script name plus a short hash of the resolved launch configuration.",
    ),
]

TimeoutOpt = Annotated[
    str | None,
    Option(
        help="Max duration: int with s (seconds, default), m (minutes), h (hours) or d (days).",
    ),
]

AttemptsOpt = Annotated[
    int | None,
    Option("--attempts", min=1, help="Maximum attempts, including the first run. Defaults to 1."),
]

DetachOpt = Annotated[
    bool,
    Option(
        "-d",
        "--detach",
        help="Run the Job in the background and print the Job ID.",
    ),
]

DryRunOpt = Annotated[
    bool,
    Option(
        "--dry-run",
        help="Print the resolved Job configuration without submitting the Job.",
    ),
]

NamespaceOpt = Annotated[
    str | None,
    Option(
        help="The namespace where the job will be running. Defaults to the current user's namespace.",
    ),
]

ResourceGroupIdOpt = Annotated[
    str | None,
    Option(
        "--resource-group-id",
        help="The ID of the resource group to create the Job in. Used to control access to resources within an organization and for cost attribution/spending-limit features.",
    ),
]

ExposeOpt = Annotated[
    list[int] | None,
    Option(
        "--expose",
        help="Expose a container port through the jobs proxy. Repeat the flag for multiple ports (e.g. `--expose 8000 --expose 8001`). Each exposed port is reachable on the public jobs domain; access requires an HF token with read access to the job's namespace.",
    ),
]

ExposePublicOpt = Annotated[
    list[int] | None,
    Option(
        "--expose-public",
        help="Expose a container port through the jobs proxy without authentication. Repeat the flag for multiple ports. No need to also pass the port with `--expose`.",
    ),
]


SshEnabledOpt = Annotated[
    bool,
    Option(
        "--ssh",
        help="Make the job's container reachable over SSH. Connect with `hf jobs ssh <job_id>`. Requires an SSH public key registered on https://huggingface.co/settings/keys.",
    ),
]

NetworkGroupOpt = Annotated[
    str | None,
    Option(
        "--network-group",
        help="Join a network group. Jobs in the same namespace and resource group sharing a group are placed together and reach each other on every port. Inside each member, `$HF_NETWORK_GROUP_HOSTNAME` resolves to every member. Lowercase alphanumerics and dashes, 46 characters max.",
    ),
]

NetworkAliasOpt = Annotated[
    list[str] | None,
    Option(
        "--network-alias",
        help="Claim an alias in the network group. Members reach the jobs claiming it at `${HF_NETWORK_GROUP_PREFIX}<alias>`. Repeat the flag for several aliases. Lowercase alphanumerics and dashes, 34 characters max, unique within the job. Requires `--network-group`.",
    ),
]

WithOpt = Annotated[
    list[str] | None,
    Option(
        "--with",
        help="Run with the given packages installed",
    ),
]

PythonOpt = Annotated[
    str | None,
    Option(
        "-p",
        "--python",
        help="The Python interpreter to use for the run environment",
    ),
]

SuspendOpt = Annotated[
    bool | None,
    Option(
        help="Suspend (pause) the scheduled Job",
    ),
]

ConcurrencyOpt = Annotated[
    bool | None,
    Option(
        help="Allow multiple instances of this Job to run concurrently",
    ),
]

ScheduleArg = Annotated[
    str,
    Argument(
        help="One of @annually, @yearly, @monthly, @weekly, @daily, @hourly, or a CRON schedule expression.",
    ),
]

ScriptArg = Annotated[
    str,
    Argument(
        help="UV script to run (local file or URL)",
    ),
]

ScriptArgsArg = Annotated[
    list[str] | None,
    Argument(
        help="Arguments for the script",
    ),
]


CommandArg = Annotated[
    list[str],
    Argument(
        help="The command to run.",
    ),
]

JobIdArg = Annotated[
    str,
    Argument(
        help="Job ID (or 'namespace/job_id')",
    ),
]

JobIdsArg = Annotated[
    list[str] | None,
    Argument(
        help="Job IDs (or 'namespace/job_id')",
    ),
]

ScheduledJobIdArg = Annotated[
    str,
    Argument(
        help="Scheduled Job ID (or 'namespace/scheduled_job_id')",
    ),
]

JobVolumesOpt = Annotated[
    list[str] | None,
    Option(
        "-v",
        "--volume",
        help="Mount one or more volumes. Format: hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] or LOCAL_DIR:/MOUNT_PATH[:ro|:rw]. "
        "TYPE is one of: models, datasets, spaces, buckets. "
        "TYPE defaults to models if omitted. "
        "models, datasets and spaces are always mounted read-only. buckets are read+write by default. "
        "A local directory source is first synced to a bucket and mounted read-only by default. "
        "E.g. -v hf://datasets/org/ds:/data or -v hf://buckets/org/b:/mnt:ro or -v ./inputs:/inputs",
    ),
]


jobs_cli = typer_factory(help="Run and manage Jobs on the Hub.")


def _stream_logs_and_check_status(api: HfApi, job: JobInfo) -> None:
    """Stream Job logs until the Job ends, then fail the command if the Job did not complete successfully."""
    for log in api.fetch_job_logs(job_id=job.id, namespace=job.owner.name, follow=True):
        out.text(log)
    # The log stream can end while the Job is still scheduling or shutting down: settle the final state.
    final = api.wait_for_job(job_id=job.id, namespace=job.owner.name)
    if final.status.stage != JobStage.COMPLETED:
        message = f": {final.status.message}" if final.status.message else ""
        raise CLIError(f"Job {final.id} finished with stage '{final.status.stage}'{message}")
    out.text(f"Job {final.id} completed")


def _hint_or_follow_started_job(api: HfApi, job: JobInfo, *, detach: bool) -> None:
    """Print hints about how to reach a started Job, then stream its logs unless `detach` is set."""
    job_ref = f"{job.owner.name}/{job.id}"
    if isinstance(job.status.expose_urls, list):
        urls = "\n".join(f"  {url}" for url in job.status.expose_urls)
        out.hint(f"Exposed ports are reachable at:\n{urls}")
    if isinstance(job.status.ssh_url, str):
        out.hint(f"Use `hf jobs ssh {job_ref}` to open an SSH session into the job.")
    if isinstance(job.network, JobNetwork):
        group = job.network.group
        out.hint(
            f"Joined network group '{group}'. Jobs of this namespace and resource group started with "
            f"`--network-group {group}` reach each other at `$HF_NETWORK_GROUP_HOSTNAME` (every member) "
            "or `${HF_NETWORK_GROUP_PREFIX}<alias>` (members claiming an alias)."
        )
    if detach:
        out.hint(f"Use `hf jobs logs -f {job_ref}` to stream logs, or `hf jobs inspect {job_ref}` to check status.")
        out.hint(f"Use `hf jobs wait {job_ref}` to block until it finishes.")
        return
    _stream_logs_and_check_status(api, job)


@jobs_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "hf jobs run --name hello-world python:3.12 python -c 'print(\"Hello!\")'",
        "hf jobs run --detach python:3.12 python script.py",
        "hf jobs run -e FOO=foo python:3.12 python script.py",
        "hf jobs run --secrets HF_TOKEN python:3.12 python script.py",
        "hf jobs run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt python:3.12 python script.py",
    ],
)
def jobs_run(
    image: ImageArg,
    command: CommandArg,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    attempts: AttemptsOpt = None,
    detach: DetachOpt = False,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    expose_public: ExposePublicOpt = None,
    ssh: SshEnabledOpt = False,
    network_group: NetworkGroupOpt = None,
    network_alias: NetworkAliasOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    format: FormatOpt = None,
    json_output: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Run a Job."""
    set_output_format(format, json_output, quiet)
    env_map = parse_env_map(env, env_file)
    secrets_map = _parse_secrets_map(secrets, secrets_file, dry_run=dry_run)
    labels_map = _parse_labels_map(label, name=name) or {}
    labels_map.setdefault(
        "name",
        _default_job_name_from_image(
            image,
            command,
            config_parts=_name_hash_parts(
                flavor=flavor,
                timeout=timeout,
                extra=[str(attempts)] if attempts is not None else None,
                namespace=namespace,
                env=env_map,
                secrets=secrets_map,
                volume_specs=volume or [],
                network_group=network_group,
                network_aliases=network_alias,
            ),
        ),
    )

    api = get_hf_api(token=token)
    volumes = None if dry_run else _parse_and_sync_job_volumes(volume, api=api, namespace=namespace)
    _print_job_summary(
        {
            "image": image,
            "command": shlex.join(command),
            "flavor": flavor or JobHardware.CPU_BASIC.value,
            "timeout": timeout,
            "attempts": attempts,
            "env": env_map,
            "secrets": secrets_map,
            "volumes": volume or [],
            "labels": labels_map,
            "expose": " ".join(str(port) for port in expose or []),
            "expose_public": " ".join(str(port) for port in expose_public or []),
            "ssh": ssh,
            "network_group": network_group,
            "network_aliases": " ".join(network_alias or []),
            "resource_group_id": resource_group_id,
            "namespace": namespace,
        },
        dry_run=dry_run,
    )
    if dry_run:
        return
    job = api.run_job(
        image=image,
        command=command,
        env=env_map,
        secrets=secrets_map,
        labels=labels_map,
        volumes=volumes,
        flavor=flavor,
        timeout=timeout,
        attempts=attempts,
        expose=expose,
        expose_public=expose_public,
        ssh=ssh,
        network_group=network_group,
        network_aliases=network_alias,
        resource_group_id=resource_group_id,
        namespace=namespace,
    )
    out.result("Job started", id=job.id, name=(job.labels or {}).get("name"), url=job.url)
    if not _has_explicit_name(name, label):
        auto_name = (job.labels or {}).get("name")
        out.hint(
            f"Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    _hint_or_follow_started_job(api, job, detach=detach)


@jobs_cli.command(
    "logs",
    examples=[
        "hf jobs logs <job_id>",
        "hf jobs logs -f <job_id>",
        "hf jobs logs --tail 20 <job_id>",
        "hf jobs logs -f --tail 100 <job_id>",
    ],
)
def jobs_logs(
    job_id: JobIdArg,
    follow: Annotated[
        bool,
        Option(
            "-f",
            "--follow",
            help="Follow log output (stream until the job completes). Without this flag, only currently available logs are printed.",
        ),
    ] = False,
    tail: Annotated[
        int | None,
        Option(
            "-n",
            "--tail",
            help="Number of lines to show from the end of the logs. When combined with --follow, starts streaming from the last N lines.",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the logs of a Job.

    By default, prints currently available logs and exits (non-blocking).
    Use --follow/-f to stream logs in real-time until the job completes.
    Use --tail/-n to limit the number of lines returned (server-side when supported).

    Note: following exits when the log stream ends, regardless of whether the Job
    succeeded or failed. Run `hf jobs inspect <job_id>` to check the final status.
    """
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)

    api = get_hf_api(token=token)
    logs = api.fetch_job_logs(job_id=job_id, namespace=namespace, follow=follow, tail=tail)
    for log in logs:
        out.text(log)
    job_ref = f"{namespace}/{job_id}" if namespace else job_id
    if follow:
        out.hint(f"Stream ended. Run `hf jobs inspect {job_ref}` to check the final status (e.g. COMPLETED or ERROR).")
    else:
        out.hint(
            f"If the job is still running, use `hf jobs logs -f {job_ref}` to stream new logs until it completes."
        )


def _clear_line(n: int) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def _format_job_stats_rows(job_id: str, metrics: dict[str, Any], table_headers: list[str]) -> list[list[str | int]]:
    row: list[str | int] = [
        job_id,
        f"{metrics['cpu_usage_pct']}%",
        round(metrics["cpu_millicores"] / 1000.0, 1),
        f"{round(100 * metrics['memory_used_bytes'] / metrics['memory_total_bytes'], 2)}%",
        f"{_format_size(metrics['memory_used_bytes'])}B / {_format_size(metrics['memory_total_bytes'])}B",
        f"{_format_size(metrics['rx_bps'])}bps / {_format_size(metrics['tx_bps'])}bps",
    ]
    if metrics["gpus"] and isinstance(metrics["gpus"], dict):
        rows: list[list[str | int]] = [row] + [[""] * len(row) for _ in range(len(metrics["gpus"]) - 1)]
        for row, gpu_id in zip(rows, sorted(metrics["gpus"])):
            gpu = metrics["gpus"][gpu_id]
            row += [
                f"{gpu['utilization']}%",
                f"{round(100 * gpu['memory_used_bytes'] / gpu['memory_total_bytes'], 2)}%",
                f"{_format_size(gpu['memory_used_bytes'])}B / {_format_size(gpu['memory_total_bytes'])}B",
            ]
        return rows
    row += ["N/A"] * (len(table_headers) - len(row))
    return [row]


def _get_jobs_stats_rows(
    job_id: str, metrics_stream: Iterable[dict[str, Any]], table_headers: list[str]
) -> Iterable[tuple[bool, str, list[list[str | int]]]]:
    for metrics in metrics_stream:
        yield False, job_id, _format_job_stats_rows(job_id, metrics, table_headers)
    yield True, job_id, []


@jobs_cli.command(
    "stats",
    examples=[
        "hf jobs stats",
        "hf jobs stats <job_id>",
        "hf jobs stats -f <job_id>",
    ],
)
def jobs_stats(
    job_ids: JobIdsArg = None,
    follow: Annotated[
        bool,
        Option(
            "-f",
            "--follow",
            help="Follow stats output (live view until the Jobs complete). Without this flag, a single snapshot is printed.",
        ),
    ] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the resource usage statistics and metrics of Jobs.

    By default, prints a snapshot of the current stats and exits (non-blocking).
    Use --follow/-f to display live stats until the Jobs complete.
    """
    if follow and out.mode != OutputFormat.human:
        raise CLIError(
            f"`--follow` displays a live terminal view and is not supported with '{out.mode.value}' output."
            " Run without `--follow` to get a snapshot."
        )
    if job_ids is not None:
        parsed_ids = []
        for job_id in job_ids:
            job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
            parsed_ids.append(job_id)
        job_ids = parsed_ids
    api = get_hf_api(token=token)
    if namespace is None:
        namespace = api.whoami()["name"]
    if job_ids is None:
        job_ids = [
            job.id
            for job in api.list_jobs(namespace=namespace)
            if (job.status.stage if job.status else "UNKNOWN") in ("RUNNING", "UPDATING")
        ]
    if len(job_ids) == 0:
        out.table([])
        out.hint("No running jobs. Use `hf jobs ps -a` to list finished (and failed) jobs.")
        return
    table_headers = [
        "JOB ID",
        "CPU %",
        "NUM CPU",
        "MEM %",
        "MEM USAGE",
        "NET I/O",
        "GPU UTIL %",
        "GPU MEM %",
        "GPU MEM USAGE",
    ]
    if not follow:

        def fetch_latest_metrics(job_id: str) -> dict[str, Any] | None:
            return next(iter(api.fetch_job_metrics(job_id=job_id, namespace=namespace)), None)

        with multiprocessing.pool.ThreadPool(len(job_ids)) as pool:
            latest_metrics = pool.map(fetch_latest_metrics, job_ids)
        items = []
        for job_id, metrics in zip(job_ids, latest_metrics):
            if metrics is None:
                out.warning(f"No stats available for Job {job_id}. Is it running?")
            else:
                items.append({"job_id": job_id, **metrics})
        if out.mode == OutputFormat.human and items:
            rows = [row for item in items for row in _format_job_stats_rows(item["job_id"], item, table_headers)]
            out.text(_tabulate(rows, headers=table_headers))
            job_refs = " ".join(f"{namespace}/{item['job_id']}" for item in items)
            out.hint(f"Use `hf jobs stats -f {job_refs}` to follow live stats.")
        else:
            out.table(items, id_key="job_id")
        return
    with multiprocessing.pool.ThreadPool(len(job_ids)) as pool:
        rows_per_job_id: dict[str, list[list[str | int]]] = {}
        for job_id in job_ids:
            row: list[str | int] = [job_id]
            row += ["-- / --" if ("/" in header or "USAGE" in header) else "--" for header in table_headers[1:]]
            rows_per_job_id[job_id] = [row]
        last_update_time = time.time()
        total_rows = [row for job_id in rows_per_job_id for row in rows_per_job_id[job_id]]
        # In-place refresh (cursor-up + clear) requires a fixed line count and layout —
        # `out.table`'s mode-dependent formatting would break it.
        print(_tabulate(total_rows, headers=table_headers))

        kwargs_list = [
            {
                "job_id": job_id,
                "metrics_stream": api.fetch_job_metrics(job_id=job_id, namespace=namespace, follow=True),
                "table_headers": table_headers,
            }
            for job_id in job_ids
        ]
        for done, job_id, rows in iflatmap_unordered(pool, _get_jobs_stats_rows, kwargs_list=kwargs_list):
            if done:
                rows_per_job_id.pop(job_id, None)
            else:
                rows_per_job_id[job_id] = rows
            now = time.time()
            if now - last_update_time >= STATS_UPDATE_MIN_INTERVAL:
                _clear_line(2 + len(total_rows))
                total_rows = [row for job_id in rows_per_job_id for row in rows_per_job_id[job_id]]
                print(_tabulate(total_rows, headers=table_headers))
                last_update_time = now
    job_refs = " ".join(f"{namespace}/{job_id}" for job_id in job_ids)
    out.hint(f"Stream ended. Run `hf jobs inspect {job_refs}` to check the final status (e.g. COMPLETED or ERROR).")


@jobs_cli.command(
    "list | ls | ps",
    examples=[
        "hf jobs ls",
        "hf jobs ls -a",
        "hf jobs ls --status running,scheduling",
        "hf jobs ls --name training-v2",
        "hf jobs ls --label env=prod --label team=ml",
        "hf jobs ls --all --label hf-sandbox=1",
    ],
)
def jobs_ps(
    all: Annotated[
        bool,
        Option(
            "-a",
            "--all",
            help="Show all Jobs (default shows running and scheduling). Cannot be combined with --status.",
        ),
    ] = False,
    status: Annotated[
        list[str] | None,
        Option(
            "--status",
            click_type=SoftChoice(JobStage),
            help="Only show Jobs with the given status. Comma-separated or repeated, e.g. `--status running,scheduling`.",
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        Option(
            "-l",
            "--label",
            help="Only show Jobs with the given `key=value` label. Repeat to require several labels, e.g. `--label env=prod --label team=ml`.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        Option(
            "--name",
            help="Only show Jobs with the given name (shortcut for `--label name=NAME`).",
        ),
    ] = None,
    limit: Annotated[
        int,
        Option(
            "--limit",
            help="Maximum number of Jobs to display. Set to 0 to show all (no limit).",
        ),
    ] = 100,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List Jobs.

    Use `--status` to filter by status (see [`JobStage`] for possible values) and `--label` to filter by `key=value`
    labels. A Job must match every filter to be listed.
    """
    api = get_hf_api(token=token)

    if all and status:
        raise CLIError("`-a`/`--all` cannot be combined with `--status`.")

    # Status filtering (default to active Jobs, unless `--all` or `--status` is provided).
    raw_statuses: list[str] = []
    for value in status or []:
        raw_statuses.extend(part.strip() for part in value.split(",") if part.strip())

    server_statuses: list[str] | None
    if raw_statuses:
        server_statuses = raw_statuses
    elif all:
        server_statuses = None
    else:
        server_statuses = [JobStage.RUNNING.value, JobStage.SCHEDULING.value]

    # Labels filtering
    labels: dict[str, str] = {}
    for item in label or []:
        if "=" not in item:
            raise CLIError(f"Invalid label filter '{item}': must be in the form 'key=value'")
        key, value = item.split("=")
        labels[key] = value

    # `--name` is a shortcut for the `name` label.
    if name is not None:
        if "name" in labels:
            raise CLIError("Cannot filter by both `--name` and `--label name=...`.")
        labels["name"] = name

    jobs_iter = api.list_jobs(namespace=namespace, status=server_statuses, labels=labels or None)

    # Apply the display limit. Fetch one extra Job to detect (and warn about) truncation.
    truncated = False
    if limit > 0:
        jobs = list(itertools.islice(jobs_iter, limit + 1))
        if len(jobs) > limit:
            truncated = True
            jobs = jobs[:limit]
    else:
        jobs = list(jobs_iter)

    # Build display items. Augment the raw api dict with curated, table-friendly columns.
    job_items: list[dict[str, Any]] = []
    for job in jobs:
        job_item = _dataclass_to_dict(job)
        durations = job_item.get("durations") or {}
        cmd = job_item.get("command") or []
        job_item["job_id"] = job_item.get("id", "")
        job_item["name"] = (job_item.get("labels") or {}).get("name") or "N/A"
        job_item["image/space"] = job_item.get("docker_image") or "N/A"
        job_item["command"] = " ".join(cmd) if cmd else "N/A"
        job_item["created"] = job_item["created_at"][:19].replace("T", " ") if job_item.get("created_at") else "N/A"
        job_item["status"] = (job_item.get("status") or {}).get("stage", "UNKNOWN")
        job_item["runtime"] = format_duration(durations.get("running_secs"))
        job_items.append(job_item)

    out.table(
        job_items,
        headers=["job_id", "name", "image/space", "command", "created", "status", "runtime"],
        id_key="job_id",
    )
    if truncated:
        out.hint(f"Output truncated to {limit} Jobs. Use `--limit 0` to show all (or `--limit N`).")
    if not job_items:
        if raw_statuses or labels:
            filters_msg = ", ".join(
                [*(f"status={s}" for s in raw_statuses), *(f"label={k}={v}" for k, v in labels.items())]
            )
            out.text(f"No jobs matched filters: {filters_msg}")
        elif not all:
            out.hint("No running jobs. Use `-a`/`--all` to include finished (and failed) jobs.")


@jobs_cli.command("hardware", examples=["hf jobs hardware"])
def jobs_hardware() -> None:
    """List available hardware options for Jobs"""
    api = get_hf_api()
    hardware_list = api.list_jobs_hardware()
    items = []
    for hw in hardware_list:
        accelerator_info = ""
        if hw.accelerator:
            accelerator_info = f"{hw.accelerator.quantity}x {hw.accelerator.model} ({hw.accelerator.vram})"
        cost_min = f"${hw.unit_cost_usd:.4f}" if hw.unit_cost_usd else "free"
        cost_hour = f"${hw.unit_cost_usd * 60:.2f}" if hw.unit_cost_usd else "free"
        items.append(
            {
                "name": hw.name,
                "pretty name": hw.pretty_name,
                "cpu": hw.cpu,
                "ram": hw.ram,
                "storage": hw.ephemeral_storage,
                "accelerator": accelerator_info,
                "cost/min": cost_min,
                "cost/hour": cost_hour,
            }
        )
    out.table(items)
    out.hint("Use `hf jobs run --flavor <name> ...` to request a specific hardware flavor.")


@jobs_cli.command("inspect", examples=["hf jobs inspect <job_id>"])
def jobs_inspect(
    job_ids: Annotated[
        list[str],
        Argument(
            help="Job IDs to inspect (or 'namespace/job_id')",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Display detailed information on one or more Jobs"""
    parsed_ids = []
    for job_id in job_ids:
        job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(job_id)
    job_ids = parsed_ids
    api = get_hf_api(token=token)
    jobs = [api.inspect_job(job_id=job_id, namespace=namespace) for job_id in job_ids]
    out.table([_surface_name(_dataclass_to_dict(job), labels=job.labels) for job in jobs], id_key="id")


@jobs_cli.command("rerun", examples=["hf jobs rerun <job_id>", "hf jobs rerun --detach <job_id>"])
def jobs_rerun(
    job_id: JobIdArg,
    detach: DetachOpt = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Run a new Job with an existing Job's spec."""
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    job = api.rerun_job(job_id=job_id, namespace=namespace)
    out.result("Job started", id=job.id, name=(job.labels or {}).get("name"), url=job.url)
    _hint_or_follow_started_job(api, job, detach=detach)


@jobs_cli.command(
    "expose",
    examples=[
        "hf jobs expose <job_id> 8000",
        "hf jobs expose <job_id> 8000 --public 9000",
        "hf jobs expose <job_id> --clear",
    ],
)
def jobs_expose(
    job_id: JobIdArg,
    ports: Annotated[
        list[int] | None,
        Argument(
            help="Ports to expose through the jobs proxy. Access requires an HF token with read access to the job's namespace."
        ),
    ] = None,
    public: Annotated[
        list[int] | None,
        Option(
            "--public",
            help="Expose a port without authentication. Repeat the flag for multiple ports. No need to also pass the port as a positional argument.",
        ),
    ] = None,
    clear: Annotated[bool, Option("--clear", help="Close all exposed ports.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Replace exposed ports on a running Job."""
    if clear and (ports or public):
        raise CLIError("`--clear` cannot be combined with ports or `--public`.")
    if not clear and not ports and not public:
        raise CLIError("Pass at least one port or `--public` port, or `--clear` to close all ports.")
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    job = api.update_job_expose(job_id=job_id, expose=ports, expose_public=public, namespace=namespace)
    out.result("Exposed ports updated", id=job.id, ports=job.expose or [], public_ports=job.expose_public or [])


@jobs_cli.command("cancel", examples=["hf jobs cancel <job_id>"])
def jobs_cancel(
    job_id: JobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Cancel a Job"""
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    api.cancel_job(job_id=job_id, namespace=namespace)
    out.result("Job cancelled", id=job_id)


@jobs_cli.command(
    "wait",
    examples=[
        "hf jobs wait <job_id>",
        "hf jobs wait <job_id_1> <job_id_2>",
        "hf jobs ls -q | xargs hf jobs wait",
    ],
)
def jobs_wait(
    job_ids: Annotated[
        list[str],
        Argument(
            help="Job IDs to wait for (or 'namespace/job_id').",
        ),
    ],
    timeout: Annotated[
        str | None,
        Option(
            help="Max time to wait: int with s (seconds, default), m (minutes), h (hours) or d (days).",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Wait for one or more Jobs to reach a terminal state.

    Blocks until every Job has finished, then exits with code 0 if all Jobs completed
    successfully, or a non-zero exit code if any Job was canceled, errored or deleted.

    All Jobs must belong to the same namespace.
    """
    parsed_ids = []
    namespaces = set()
    for job_id in job_ids:
        parsed_id, parsed_namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(parsed_id)
        namespaces.add(parsed_namespace)
    if len(namespaces) > 1:
        raise CLIError(
            "All Job IDs must be in the same namespace, got: "
            + ", ".join(str(ns) for ns in sorted(namespaces, key=str))
        )
    namespace = namespaces.pop()
    timeout_secs = parse_duration(timeout) if timeout is not None else None

    api = get_hf_api(token=token)
    status = out.status(f"Waiting for {len(parsed_ids)} Job(s) to finish...")
    try:
        jobs = api.wait_for_job(parsed_ids, timeout=timeout_secs, namespace=namespace)
    except TimeoutError:
        status.done("Timed out.")
        raise CLIError(f"Timed out after {timeout} waiting for Job(s) to finish.") from None
    status.done(f"{len(jobs)} Job(s) finished.")

    out.table([{"id": job.id, "stage": str(job.status.stage), "message": job.status.message} for job in jobs])
    failed = [job for job in jobs if job.status.stage != JobStage.COMPLETED]
    if failed:
        raise CLIError(
            f"{len(failed)} of {len(jobs)} Job(s) did not complete successfully: "
            + ", ".join(f"{job.id} ({job.status.stage})" for job in failed)
        )


@jobs_cli.command(
    "labels",
    examples=[
        "hf jobs labels <job_id> --name training-v2",
        "hf jobs labels <job_id> --label env=prod --label team=ml",
        "hf jobs labels <job_id> --clear",
    ],
)
def jobs_labels(
    job_id: JobIdArg,
    name: NameOpt = None,
    label: LabelsOpt = None,
    clear: Annotated[bool, Option("--clear", help="Remove all labels from the job.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Update labels on a Job. Passing --label replaces all existing labels; passing --name alone keeps them."""
    if not label and name is None and not clear:
        raise CLIError(
            "Please set a name with --name or at least one label with --label. To remove all labels, pass --clear."
        )
    if (label or name is not None) and clear:
        raise CLIError(
            "Cannot set a name or labels and clear them at the same time. Please use --name/--label or --clear, not both."
        )
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    if name is not None and not label:
        # Naming a Job should not wipe its existing labels: fetch them and merge the name in.
        current_labels = api.inspect_job(job_id=job_id, namespace=namespace).labels or {}
        labels = {**current_labels, "name": name}
    else:
        labels = _parse_labels_map(label, name=name) or {}
    job = api.update_job_labels(job_id=job_id, labels=labels, namespace=namespace)
    out.result("Labels updated", id=job.id, name=labels.get("name"))


@jobs_cli.command(
    "ssh",
    examples=[
        "hf jobs ssh <job_id>",
        "hf jobs ssh <job_id> --dry-run",
        "hf jobs ssh <job_id> -i ~/.ssh/id_ed25519",
    ],
)
def jobs_ssh(
    job_id: JobIdArg,
    identity_file: SshIdentityFileOpt = None,
    dry_run: SshDryRunOpt = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """SSH into a running Job.

    If the Job is not yet running, waits until it reaches the RUNNING state before
    connecting. Requires the Job to be started with SSH enabled (`hf jobs run --ssh ...`)
    and your SSH public key to be registered at https://huggingface.co/settings/keys.
    """
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    job = api.inspect_job(job_id=job_id, namespace=namespace)
    if job.status.ssh_url is None:
        raise CLIError("SSH is not enabled on this job. Start a job with SSH support using `hf jobs run --ssh ...`.")
    if job.status.stage in TERMINAL_JOB_STAGES:
        raise CLIError(f"Cannot SSH into job '{job.id}': job has already finished (stage: '{job.status.stage}').")
    if job.status.stage != JobStage.RUNNING:
        status = out.status(f"Waiting for job '{job.id}' to be running (stage: '{job.status.stage}')...")
        job = api.wait_for_job(job_id=job.id, namespace=namespace, stages=[JobStage.RUNNING])
        if job.status.stage != JobStage.RUNNING:
            status.done("Job finished.")
            raise CLIError(
                f"Cannot SSH into job '{job.id}': job finished before reaching RUNNING (stage: '{job.status.stage}')."
            )
        status.done("Job is running.")
    ssh_url = urlsplit(job.status.ssh_url)
    exec_ssh(
        f"{ssh_url.username}@{ssh_url.hostname}",  # type: ignore
        port=ssh_url.port,
        identity_file=identity_file,
        dry_run=dry_run,
    )


uv_app = typer_factory(help="Run UV scripts (Python with inline dependencies) on HF infrastructure.")
jobs_cli.add_group(uv_app, name="uv")


@uv_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "hf jobs uv run --name my-script my_script.py",
        "hf jobs uv run --detach my_script.py",
        "hf jobs uv run --flavor a10g-small ml_training.py",
        "hf jobs uv run --with transformers train.py",
        "hf jobs uv run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt script.py",
        "hf jobs uv run --dry-run script.py",
    ],
)
def jobs_uv_run(
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    image: ImageOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    attempts: AttemptsOpt = None,
    detach: DetachOpt = False,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    expose_public: ExposePublicOpt = None,
    ssh: SshEnabledOpt = False,
    network_group: NetworkGroupOpt = None,
    network_alias: NetworkAliasOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
    format: FormatOpt = None,
    json_output: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Run a UV script (local file or URL) on HF infrastructure"""
    set_output_format(format, json_output, quiet)
    api = get_hf_api(token=token)
    with _resolve_uv_job_config(
        api=api,
        script=script,
        script_args=script_args,
        dependencies=with_,
        image=image,
        flavor=flavor,
        python=python,
        env=env,
        env_file=env_file,
        secrets=secrets,
        secrets_file=secrets_file,
        timeout=timeout,
        attempts=attempts,
        name=name,
        label=label,
        volume=volume,
        namespace=namespace,
        network_group=network_group,
        network_aliases=network_alias,
        dry_run=dry_run,
    ) as config:
        _print_job_summary(
            {
                "script": script,
                "args": shlex.join(config.script_args),
                "with": " ".join(with_ or []),
                "image": config.image or DEFAULT_UV_IMAGE,
                "flavor": config.flavor or JobHardware.CPU_BASIC.value,
                "python": config.python,
                "timeout": config.timeout,
                "attempts": attempts,
                "env": config.env,
                "secrets": config.secrets,
                "volumes": config.volume_specs,
                "labels": config.labels,
                "expose": " ".join(str(port) for port in expose or []),
                "expose_public": " ".join(str(port) for port in expose_public or []),
                "ssh": ssh,
                "network_group": config.network_group,
                "network_aliases": " ".join(config.network_aliases),
                "resource_group_id": resource_group_id,
                "namespace": config.namespace,
            },
            from_script=config.from_script,
            dry_run=dry_run,
        )
        if dry_run:
            return
        job = api.run_uv_job(
            script=config.script,
            script_args=config.script_args,
            dependencies=with_,
            python=config.python,
            image=config.image,
            env=config.env,
            secrets=config.secrets,
            labels=config.labels,
            volumes=config.volumes,
            flavor=config.flavor,
            timeout=config.timeout,
            attempts=attempts,
            expose=expose,
            expose_public=expose_public,
            ssh=ssh,
            network_group=config.network_group,
            network_aliases=config.network_aliases or None,
            resource_group_id=resource_group_id,
            namespace=config.namespace,
        )
    out.result("Job started", id=job.id, name=(job.labels or {}).get("name"), url=job.url)
    if not _has_explicit_name(name, label, config.from_script):
        auto_name = (job.labels or {}).get("name")
        out.hint(
            f"Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    _hint_or_follow_started_job(api, job, detach=detach)


class ScheduledJobStatusFilter(str, Enum):
    """Possible values for `hf jobs scheduled ls --status`.

    Scheduled Jobs are not "running": they are either active (i.e. they will trigger new runs) or suspended.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"


scheduled_app = typer_factory(help="Create and manage scheduled Jobs on the Hub.")
jobs_cli.add_group(scheduled_app, name="scheduled")


@scheduled_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=['hf jobs scheduled run "0 0 * * *" --name daily-script python:3.12 python script.py'],
)
def scheduled_run(
    schedule: ScheduleArg,
    image: ImageArg,
    command: CommandArg,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    attempts: AttemptsOpt = None,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    expose_public: ExposePublicOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    format: FormatOpt = None,
    json_output: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Schedule a Job."""
    set_output_format(format, json_output, quiet)
    env_map = parse_env_map(env, env_file)
    secrets_map = _parse_secrets_map(secrets, secrets_file, dry_run=dry_run)
    labels_map = _parse_labels_map(label, name=name) or {}
    labels_map.setdefault(
        "name",
        _default_job_name_from_image(
            image,
            command,
            config_parts=_name_hash_parts(
                flavor=flavor,
                timeout=timeout,
                extra=[str(attempts)] if attempts is not None else None,
                namespace=namespace,
                env=env_map,
                secrets=secrets_map,
                volume_specs=volume or [],
            ),
        ),
    )

    api = get_hf_api(token=token)
    volumes = None if dry_run else _parse_and_sync_job_volumes(volume, api=api, namespace=namespace)
    _print_job_summary(
        {
            "schedule": schedule,
            "suspend": suspend,
            "concurrency": concurrency,
            "image": image,
            "command": shlex.join(command),
            "flavor": flavor or JobHardware.CPU_BASIC.value,
            "timeout": timeout,
            "attempts": attempts,
            "env": env_map,
            "secrets": secrets_map,
            "volumes": volume or [],
            "labels": labels_map,
            "expose": " ".join(str(port) for port in expose or []),
            "expose_public": " ".join(str(port) for port in expose_public or []),
            "resource_group_id": resource_group_id,
            "namespace": namespace,
        },
        dry_run=dry_run,
    )
    if dry_run:
        return
    scheduled_job = api.create_scheduled_job(
        image=image,
        command=command,
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        env=env_map,
        secrets=secrets_map,
        labels=labels_map,
        volumes=volumes,
        flavor=flavor,
        timeout=timeout,
        attempts=attempts,
        expose=expose,
        expose_public=expose_public,
        resource_group_id=resource_group_id,
        namespace=namespace,
    )
    out.result("Scheduled Job created", id=scheduled_job.id, name=(scheduled_job.job_spec.labels or {}).get("name"))
    if not _has_explicit_name(name, label):
        auto_name = (scheduled_job.job_spec.labels or {}).get("name")
        out.hint(
            f"Scheduled Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs scheduled labels {scheduled_job.owner.name}/{scheduled_job.id} --name NAME` to rename it."
        )
    out.hint(f"Use `hf jobs scheduled inspect {scheduled_job.owner.name}/{scheduled_job.id}` to view its details.")


@scheduled_app.command(
    "list | ls | ps",
    examples=[
        "hf jobs scheduled ls",
        "hf jobs scheduled ls -a",
        "hf jobs scheduled ls --status suspended",
        "hf jobs scheduled ls --name daily-script",
        "hf jobs scheduled ls --label env=prod --label team=ml",
    ],
)
def scheduled_ps(
    all: Annotated[
        bool,
        Option(
            "-a",
            "--all",
            help="Show all scheduled Jobs (default hides suspended). Cannot be combined with --status.",
        ),
    ] = False,
    status: Annotated[
        list[str] | None,
        Option(
            "--status",
            click_type=SoftChoice(ScheduledJobStatusFilter),
            help=(
                "Only show scheduled Jobs with the given status. Comma-separated or repeated, e.g."
                " `--status suspended`."
            ),
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        Option(
            "-l",
            "--label",
            help=(
                "Only show scheduled Jobs with the given `key=value` label. Repeat to require several labels, e.g."
                " `--label env=prod --label team=ml`."
            ),
        ),
    ] = None,
    name: Annotated[
        str | None,
        Option(
            "--name",
            help="Only show scheduled Jobs with the given name (shortcut for `--label name=NAME`).",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List scheduled Jobs.

    Use `--status` to filter by status (`active` or `suspended`) and `--label` to filter by `key=value` labels.
    A scheduled Job must match every filter to be listed.
    """
    api = get_hf_api(token=token)

    if all and status:
        raise CLIError("`-a`/`--all` cannot be combined with `--status`.")

    # Status filtering (default to active scheduled Jobs, unless `--all` or `--status` is provided).
    raw_statuses: list[str] = []
    for value in status or []:
        raw_statuses.extend(part.strip().lower() for part in value.split(",") if part.strip())

    unknown_statuses = [s for s in raw_statuses if s not in tuple(ScheduledJobStatusFilter)]
    if unknown_statuses:
        raise CLIError(
            f"Invalid status filter(s) {unknown_statuses}: expected one of"
            f" {[s.value for s in ScheduledJobStatusFilter]}."
        )

    if raw_statuses:
        show_active = ScheduledJobStatusFilter.ACTIVE in raw_statuses
        show_suspended = ScheduledJobStatusFilter.SUSPENDED in raw_statuses
    else:
        show_active = True
        show_suspended = all

    # Labels filtering
    labels: dict[str, str] = {}
    for raw_label in label or []:
        if "=" not in raw_label:
            raise CLIError(f"Invalid label filter '{raw_label}': must be in the form 'key=value'")
        key, value = raw_label.split("=", 1)
        labels[key] = value

    # `--name` is a shortcut for the `name` label.
    if name is not None:
        if "name" in labels:
            raise CLIError("Cannot filter by both `--name` and `--label name=...`.")
        labels["name"] = name

    scheduled_jobs = api.list_scheduled_jobs(namespace=namespace, labels=labels or None)

    filtered_jobs = []
    for scheduled_job in scheduled_jobs:
        suspend = scheduled_job.suspend or False
        if suspend and not show_suspended:
            continue
        if not suspend and not show_active:
            continue
        filtered_jobs.append(scheduled_job)

    # Build display items. Augment with curated columns.
    items: list[dict[str, Any]] = []
    for sj in filtered_jobs:
        item = _dataclass_to_dict(sj)
        job_spec = item.get("job_spec") or {}
        status_dict = item.get("status") or {}
        last_job = status_dict.get("last_job")
        cmd = job_spec.get("command") or []
        item["name"] = (job_spec.get("labels") or {}).get("name") or "N/A"
        item["image/space"] = job_spec.get("docker_image") or "N/A"
        item["command"] = " ".join(cmd) if cmd else "N/A"
        item["last_run"] = last_job["at"][:19].replace("T", " ") if last_job and last_job.get("at") else "N/A"
        item["next_run"] = (
            status_dict["next_job_run_at"][:19].replace("T", " ") if status_dict.get("next_job_run_at") else "N/A"
        )
        item["suspend"] = item.get("suspend") or False
        items.append(item)

    out.table(
        items,
        headers=["id", "name", "schedule", "image/space", "command", "last_run", "next_run", "suspend"],
        id_key="id",
    )
    if not items:
        if raw_statuses or labels:
            filters_msg = ", ".join(
                [*(f"status={s}" for s in raw_statuses), *(f"label={k}={v}" for k, v in labels.items())]
            )
            out.text(f"No scheduled jobs matched filters: {filters_msg}")
        elif not all:
            out.hint("No active scheduled jobs. Use `-a`/`--all` to include suspended ones.")
    if items:
        first_item_id = items[0]["id"]
        out.hint(f"Use `hf jobs scheduled inspect {first_item_id}` to view details about a scheduled job.")
        out.hint(f"Use `hf jobs scheduled trigger {first_item_id}` to trigger a scheduled job immediately.")


@scheduled_app.command("inspect", examples=["hf jobs scheduled inspect <id>"])
def scheduled_inspect(
    scheduled_job_ids: Annotated[
        list[str],
        Argument(
            help="Scheduled Job IDs to inspect (or 'namespace/scheduled_job_id')",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Display detailed information on one or more scheduled Jobs"""
    parsed_ids = []
    for job_id in scheduled_job_ids:
        job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(job_id)
    scheduled_job_ids = parsed_ids
    api = get_hf_api(token=token)
    scheduled_jobs = [
        api.inspect_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
        for scheduled_job_id in scheduled_job_ids
    ]
    out.table([_surface_name(_dataclass_to_dict(sj), labels=sj.job_spec.labels) for sj in scheduled_jobs], id_key="id")


@scheduled_app.command("delete", examples=["hf jobs scheduled delete <id>"])
def scheduled_delete(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Delete a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.delete_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job deleted", id=scheduled_job_id)


@scheduled_app.command("suspend", examples=["hf jobs scheduled suspend <id>"])
def scheduled_suspend(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Suspend (pause) a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.suspend_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job suspended", id=scheduled_job_id)
    out.hint(f"Use `hf jobs scheduled resume {scheduled_job_id}` to resume it.")


@scheduled_app.command("resume", examples=["hf jobs scheduled resume <id>"])
def scheduled_resume(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Resume (unpause) a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.resume_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job resumed", id=scheduled_job_id)


@scheduled_app.command("reschedule", examples=['hf jobs scheduled reschedule <id> "0 9 * * 1"'])
def scheduled_reschedule(
    scheduled_job_id: ScheduledJobIdArg,
    schedule: ScheduleArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Change when a scheduled Job runs."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    job = api.update_scheduled_job_schedule(scheduled_job_id=scheduled_job_id, schedule=schedule, namespace=namespace)
    out.result("Scheduled Job rescheduled", id=job.id, schedule=job.schedule)
    out.hint(f"Use `hf jobs scheduled inspect {job.owner.name}/{job.id}` to see its next run.")


@scheduled_app.command("trigger", examples=["hf jobs scheduled trigger <id>"])
def scheduled_trigger(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Trigger a scheduled Job to run immediately (does not change the schedule)."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    job = api.trigger_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job triggered", id=job.id, url=job.url)
    out.hint(f"Use `hf jobs logs -f {job.owner.name}/{job.id}` to stream logs.")


@scheduled_app.command(
    "labels",
    examples=[
        "hf jobs scheduled labels <id> --name daily-script",
        "hf jobs scheduled labels <id> --label env=prod --label team=ml",
        "hf jobs scheduled labels <id> --clear",
    ],
)
def scheduled_labels(
    scheduled_job_id: ScheduledJobIdArg,
    name: NameOpt = None,
    label: LabelsOpt = None,
    clear: Annotated[bool, Option("--clear", help="Remove all labels from the scheduled job.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Update labels on a scheduled Job. Passing --label replaces all existing labels; passing --name alone keeps them."""
    if not label and name is None and not clear:
        raise CLIError(
            "Please set a name with --name or at least one label with --label. To remove all labels, pass --clear."
        )
    if (label or name is not None) and clear:
        raise CLIError(
            "Cannot set a name or labels and clear them at the same time. Please use --name/--label or --clear, not both."
        )
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    if name is not None and not label:
        # Naming a scheduled Job should not wipe its existing labels: fetch them and merge the name in.
        current_labels = (
            api.inspect_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace).job_spec.labels or {}
        )
        labels = {**current_labels, "name": name}
    else:
        labels = _parse_labels_map(label, name=name) or {}
    scheduled_job = api.update_scheduled_job_labels(
        scheduled_job_id=scheduled_job_id, labels=labels, namespace=namespace
    )
    out.result("Labels updated", id=scheduled_job.id, name=labels.get("name"))


scheduled_uv_app = typer_factory(help="Schedule UV scripts on HF infrastructure.")
scheduled_app.add_group(scheduled_uv_app, name="uv")


@scheduled_uv_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        'hf jobs scheduled uv run "0 0 * * *" --name daily-script script.py',
        'hf jobs scheduled uv run "0 0 * * *" script.py --with pandas',
    ],
)
def scheduled_uv_run(
    schedule: ScheduleArg,
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    image: ImageOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    attempts: AttemptsOpt = None,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    expose_public: ExposePublicOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
    format: FormatOpt = None,
    json_output: JsonOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Run a UV script (local file or URL) on HF infrastructure"""
    set_output_format(format, json_output, quiet)
    api = get_hf_api(token=token)
    with _resolve_uv_job_config(
        api=api,
        script=script,
        script_args=script_args,
        dependencies=with_,
        image=image,
        flavor=flavor,
        python=python,
        env=env,
        env_file=env_file,
        secrets=secrets,
        secrets_file=secrets_file,
        timeout=timeout,
        attempts=attempts,
        name=name,
        label=label,
        volume=volume,
        namespace=namespace,
        # Scheduled Jobs have no network group: a script asking for one is an error, not a silent drop.
        network_group=None,
        network_aliases=None,
        dry_run=dry_run,
    ) as config:
        if config.network_group or config.network_aliases:
            raise CLIError(
                f"Scheduled Jobs do not support network groups: remove 'network_group'/'network_aliases' from the"
                f" script's [{TABLE_NAME}] table to schedule it."
            )
        _print_job_summary(
            {
                "schedule": schedule,
                "suspend": suspend,
                "concurrency": concurrency,
                "script": script,
                "args": shlex.join(config.script_args),
                "with": " ".join(with_ or []),
                "image": config.image or DEFAULT_UV_IMAGE,
                "flavor": config.flavor or JobHardware.CPU_BASIC.value,
                "python": config.python,
                "timeout": config.timeout,
                "attempts": attempts,
                "env": config.env,
                "secrets": config.secrets,
                "volumes": config.volume_specs,
                "labels": config.labels,
                "expose": " ".join(str(port) for port in expose or []),
                "expose_public": " ".join(str(port) for port in expose_public or []),
                "resource_group_id": resource_group_id,
                "namespace": config.namespace,
            },
            from_script=config.from_script,
            dry_run=dry_run,
        )
        if dry_run:
            return
        job = api.create_scheduled_uv_job(
            script=config.script,
            script_args=config.script_args,
            schedule=schedule,
            suspend=suspend,
            concurrency=concurrency,
            dependencies=with_,
            python=config.python,
            image=config.image,
            env=config.env,
            secrets=config.secrets,
            labels=config.labels,
            volumes=config.volumes,
            flavor=config.flavor,
            timeout=config.timeout,
            attempts=attempts,
            expose=expose,
            expose_public=expose_public,
            resource_group_id=resource_group_id,
            namespace=config.namespace,
        )
    out.result("Scheduled Job created", id=job.id, name=(job.job_spec.labels or {}).get("name"))
    if not _has_explicit_name(name, label, config.from_script):
        auto_name = (job.job_spec.labels or {}).get("name")
        out.hint(
            f"Scheduled Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs scheduled labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    out.hint(f"Use `hf jobs scheduled inspect {job.owner.name}/{job.id}` to view its details.")


### UTILS


def _surface_name(item: dict[str, Any], *, labels: dict[str, str] | None) -> dict[str, Any]:
    """Promote the `name` label to a top-level `name` field for display.

    The `name` is kept inside `labels` too (the dataclasses and API payloads are unchanged); this
    only makes it a first-class column/field in command outputs. No-op when there is no name.
    """
    name = (labels or {}).get("name")
    if name is None:
        return item
    return {"name": name, **item}


def _has_explicit_name(name: str | None, label: list[str] | None, from_script: Collection[str] = ()) -> bool:
    """Whether the Job was named on purpose: via `--name`, a `name=` label, or the script's `[tool.hf-jobs]` table."""
    return (
        name is not None
        or "labels.name" in from_script
        or any(item.split("=", 1)[0] == "name" for item in label or [])
    )


def _parse_labels_map(labels: list[str] | None, *, name: str | None = None) -> dict[str, str] | None:
    """Parse label key-value pairs from CLI arguments.

    Args:
        labels: List of label strings in KEY=VALUE format. If KEY only, then VALUE is set to empty string.

    Returns:
        Dictionary mapping label keys to values, or None if no labels or name provided.
    """
    if not labels and name is None:
        return None
    labels_map: dict[str, str] = {}
    for label_var in labels or []:
        key, value = label_var.split("=", 1) if "=" in label_var else (label_var, "")
        labels_map[key] = value
    if name is not None:
        if "name" in labels_map:
            raise CLIError("--name and --label name=... cannot both be provided.")
        labels_map["name"] = name
    return labels_map


def _tabulate(rows: list[list[str | int]], headers: list[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    terminal_width = max(shutil.get_terminal_size().columns, len(headers) * 12)
    while len(headers) + sum(col_widths) > terminal_width:
        col_to_minimize = col_widths.index(max(col_widths))
        col_widths[col_to_minimize] //= 2
        if len(headers) + sum(col_widths) <= terminal_width:
            col_widths[col_to_minimize] = terminal_width - sum(col_widths) - len(headers) + col_widths[col_to_minimize]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        row_format_args = [
            str(x)[: col_width - 3] + "..." if len(str(x)) > col_width else str(x)
            for x, col_width in zip(row, col_widths)
        ]
        lines.append(row_format.format(*row_format_args))
    return "\n".join(lines)


T = TypeVar("T")


def _write_generator_to_queue(queue: Queue[T], func: Callable[..., Iterable[T]], kwargs: dict) -> None:
    for result in func(**kwargs):
        queue.put(result)


def iflatmap_unordered(
    pool: multiprocessing.pool.ThreadPool,
    func: Callable[..., Iterable[T]],
    *,
    kwargs_list: list[dict],
) -> Iterable[T]:
    """
    Takes a function that returns an iterable of items, and run it in parallel using threads to return the flattened iterable of items as they arrive.

    This is inspired by those three `map()` variants, and is the mix of all three:

    * `imap()`: like `map()` but returns an iterable instead of a list of results
    * `imap_unordered()`: like `imap()` but the output is sorted by time of arrival
    * `flatmap()`: like `map()` but given a function which returns a list, `flatmap()` returns the flattened list that is the concatenation of all the output lists
    """
    queue: Queue[T] = Queue()
    async_results = [pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_list]
    try:
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready() for async_result in async_results) and queue.empty():
                    break
    except KeyboardInterrupt:
        pass
    finally:
        # we get the result in case there's an error to raise
        try:
            [async_result.get(timeout=0.05) for async_result in async_results]
        except multiprocessing.TimeoutError:
            pass
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
"""Contains commands to interact with collections on the Hugging Face Hub."""

import enum
from typing import Annotated, get_args

from huggingface_hub.hf_api import CollectionItemType_T, CollectionSort_T

from ._cli_utils import LimitOpt, TokenOpt, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


# Build enums dynamically from Literal types to avoid duplication
_COLLECTION_ITEM_TYPES = get_args(CollectionItemType_T)
CollectionItemType = enum.Enum("CollectionItemType", {t: t for t in _COLLECTION_ITEM_TYPES}, type=str)  # type: ignore[misc]

_COLLECTION_SORT_OPTIONS = get_args(CollectionSort_T)
CollectionSort = enum.Enum("CollectionSort", {s: s for s in _COLLECTION_SORT_OPTIONS}, type=str)  # type: ignore[misc]


collections_cli = typer_factory(help="Interact with collections on the Hub.")


@collections_cli.command(
    "list | ls",
    examples=[
        "hf collections ls",
        "hf collections ls --owner nvidia",
        "hf collections ls --item models/teknium/OpenHermes-2.5-Mistral-7B --limit 10",
    ],
)
def collections_ls(
    owner: Annotated[
        str | None,
        Option(help="Filter by owner username or organization."),
    ] = None,
    item: Annotated[
        str | None,
        Option(
            help='Filter collections containing a specific item (e.g., "models/gpt2", "datasets/squad", "papers/2311.12983").'
        ),
    ] = None,
    sort: Annotated[
        CollectionSort | None,
        Option(help="Sort results by last modified, trending, or upvotes."),
    ] = None,
    limit: LimitOpt = 10,
    token: TokenOpt = None,
) -> None:
    """List collections on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        _dataclass_to_dict(collection)
        for collection in api.list_collections(
            owner=owner,
            item=item,
            sort=sort_key,  # type: ignore[arg-type]
            limit=limit,
        )
    ]
    out.table(results)


@collections_cli.command(
    "info",
    examples=[
        "hf collections info username/my-collection-slug",
    ],
)
def collections_info(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    token: TokenOpt = None,
) -> None:
    """Get info about a collection on the Hub."""
    api = get_hf_api(token=token)
    collection = api.get_collection(collection_slug)
    out.dict(collection)


@collections_cli.command(
    "create",
    examples=[
        'hf collections create "My Models"',
        'hf collections create "My Models" --description "A collection of my favorite models" --private',
        'hf collections create "Org Collection" --namespace my-org',
    ],
)
def collections_create(
    title: Annotated[str, Argument(help="The title of the collection.")],
    namespace: Annotated[
        str | None,
        Option(help="The namespace (username or organization). Defaults to the authenticated user."),
    ] = None,
    description: Annotated[
        str | None,
        Option(help="A description for the collection (max 150 characters)."),
    ] = None,
    private: Annotated[
        bool,
        Option(help="Create a private collection."),
    ] = False,
    exists_ok: Annotated[
        bool,
        Option(help="Do not raise an error if the collection already exists."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Create a new collection on the Hub."""
    api = get_hf_api(token=token)
    collection = api.create_collection(
        title=title,
        namespace=namespace,
        description=description,
        private=private,
        exists_ok=exists_ok,
    )
    out.result("Collection created", slug=collection.slug, url=collection.url)


@collections_cli.command(
    "update",
    examples=[
        'hf collections update username/my-collection --title "New Title"',
        'hf collections update username/my-collection --description "Updated description"',
        "hf collections update username/my-collection --private --theme green",
    ],
)
def collections_update(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    title: Annotated[
        str | None,
        Option(help="The new title for the collection."),
    ] = None,
    description: Annotated[
        str | None,
        Option(help="The new description for the collection (max 150 characters)."),
    ] = None,
    position: Annotated[
        int | None,
        Option(help="The new position of the collection in the owner's list."),
    ] = None,
    private: Annotated[
        bool | None,
        Option(help="Whether the collection should be private."),
    ] = None,
    theme: Annotated[
        str | None,
        Option(help="The theme color for the collection (e.g., 'green', 'blue')."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update a collection's metadata on the Hub."""
    api = get_hf_api(token=token)
    collection = api.update_collection_metadata(
        collection_slug=collection_slug,
        title=title,
        description=description,
        position=position,
        private=private,
        theme=theme,
    )
    out.result("Collection updated", slug=collection.slug, url=collection.url)


@collections_cli.command(
    "delete",
    examples=[
        "hf collections delete username/my-collection",
        "hf collections delete username/my-collection --missing-ok",
    ],
)
def collections_delete(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    missing_ok: Annotated[
        bool,
        Option(help="Do not raise an error if the collection doesn't exist."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a collection from the Hub."""
    api = get_hf_api(token=token)
    api.delete_collection(collection_slug, missing_ok=missing_ok)
    out.result("Collection deleted", slug=collection_slug)


@collections_cli.command(
    "add-item",
    examples=[
        "hf collections add-item username/my-collection moonshotai/kimi-k2 model",
        'hf collections add-item username/my-collection Qwen/DeepPlanning dataset --note "Useful dataset"',
        "hf collections add-item username/my-collection Tongyi-MAI/Z-Image space",
    ],
)
def collections_add_item(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    item_id: Annotated[str, Argument(help="The ID of the item to add (repo_id for repos, paper ID for papers).")],
    item_type: Annotated[
        CollectionItemType,
        Argument(help="The type of item (model, dataset, space, paper, collection, or bucket)."),
    ],
    note: Annotated[
        str | None,
        Option(help="A note to attach to the item (max 500 characters)."),
    ] = None,
    exists_ok: Annotated[
        bool,
        Option(help="Do not raise an error if the item is already in the collection."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Add an item to a collection."""
    api = get_hf_api(token=token)
    collection = api.add_collection_item(
        collection_slug=collection_slug,
        item_id=item_id,
        item_type=item_type.value,  # type: ignore[arg-type]
        note=note,
        exists_ok=exists_ok,
    )
    out.result("Item added to collection", slug=collection_slug, url=collection.url)


@collections_cli.command(
    "update-item",
    examples=[
        'hf collections update-item username/my-collection ITEM_OBJECT_ID --note "Updated note"',
        "hf collections update-item username/my-collection ITEM_OBJECT_ID --position 0",
    ],
)
def collections_update_item(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    item_object_id: Annotated[
        str,
        Argument(help="The ID of the item in the collection (from 'item_object_id' field, not the repo_id)."),
    ],
    note: Annotated[
        str | None,
        Option(help="A new note for the item (max 500 characters)."),
    ] = None,
    position: Annotated[
        int | None,
        Option(help="The new position of the item in the collection."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update an item in a collection."""
    api = get_hf_api(token=token)
    api.update_collection_item(
        collection_slug=collection_slug,
        item_object_id=item_object_id,
        note=note,
        position=position,
    )
    out.result("Item updated in collection", slug=collection_slug)


@collections_cli.command("delete-item")
def collections_delete_item(
    collection_slug: Annotated[str, Argument(help="The collection slug (e.g., 'username/collection-slug').")],
    item_object_id: Annotated[
        str,
        Argument(
            help="The ID of the item in the collection (retrieved from `item_object_id` field returned by 'hf collections info'."
        ),
    ],
    missing_ok: Annotated[
        bool,
        Option(help="Do not raise an error if the item doesn't exist."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete an item from a collection."""
    api = get_hf_api(token=token)
    api.delete_collection_item(
        collection_slug=collection_slug,
        item_object_id=item_object_id,
        missing_ok=missing_ok,
    )
    out.result("Item deleted from collection", slug=collection_slug)
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
#
# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/huggingface.js/tree/main/packages/inference/src
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869
#    https://github.com/huggingface/unity-api#tasks
#
# Some TODO:
# - add all tasks
#
# NOTE: the philosophy of this client is "let's make it as easy as possible to use it, even if less optimized". Some
# examples of how it translates:
# - Timeout / Server unavailable is handled by the client in a single "timeout" parameter.
# - Files can be provided as bytes, file paths, or URLs and the client will try to "guess" the type.
# - Images are parsed as PIL.Image for easier manipulation.
# - Provides a "recommended model" for each task => suboptimal but user-wise quicker to get a first script running.
# - Only the main parameters are publicly exposed. Power users can always read the docs for more options.
import base64
import logging
import os
import re
import warnings
from collections.abc import Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

from huggingface_hub import constants
from huggingface_hub.errors import BadRequestError, HfHubHTTPError, InferenceTimeoutError
from huggingface_hub.inference._common import (
    TASKS_EXPECTING_IMAGES,
    ContentT,
    RequestParameters,
    _b64_encode,
    _b64_to_image,
    _bytes_to_dict,
    _bytes_to_image,
    _bytes_to_list,
    _get_unsupported_text_generation_kwargs,
    _import_numpy,
    _set_unsupported_text_generation_kwargs,
    _stream_chat_completion_response,
    _stream_text_generation_response,
    raise_text_generation_error,
)
from huggingface_hub.inference._generated.types import (
    AudioClassificationOutputElement,
    AudioClassificationOutputTransform,
    AudioToAudioOutputElement,
    AutomaticSpeechRecognitionOutput,
    ChatCompletionInputGrammarType,
    ChatCompletionInputMessage,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageClassificationOutputTransform,
    ImageSegmentationOutputElement,
    ImageSegmentationSubtask,
    ImageToImageTargetSize,
    ImageToTextOutput,
    ImageToVideoTargetSize,
    ObjectDetectionOutputElement,
    Padding,
    QuestionAnsweringOutputElement,
    SummarizationOutput,
    SummarizationTruncationStrategy,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TextClassificationOutputTransform,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    TextGenerationStreamOutput,
    TextToSpeechEarlyStoppingEnum,
    TokenClassificationAggregationStrategy,
    TokenClassificationOutputElement,
    TranslationOutput,
    TranslationTruncationStrategy,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    ZeroShotImageClassificationOutputElement,
)
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T, get_provider_helper
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    validate_hf_hub_args,
)


if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

logger = logging.getLogger(__name__)


MODEL_KWARGS_NOT_USED_REGEX = re.compile(r"The following `model_kwargs` are not used by the model: \[(.*?)\]")


class InferenceClient:
    """
    Initialize a new Inference Client.

    [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
    seamlessly with either the (free) Inference API, self-hosted Inference Endpoints, or third-party Inference Providers.

    Args:
        model (`str`, `optional`):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `meta-llama/Meta-Llama-3-8B-Instruct`
            or a URL to a deployed Inference Endpoint. Defaults to None, in which case a recommended model is
            automatically selected for the task.
            Note: for better compatibility with OpenAI's client, `model` has been aliased as `base_url`. Those 2
            arguments are mutually exclusive. If a URL is passed as `model` or `base_url` for chat completion, the `(/v1)/chat/completions` suffix path will be appended to the URL.
        provider (`str`, *optional*):
            Name of the provider to use for inference. Can be `"baseten"`, `"cerebras"`, `"cohere"`, `"deepinfra"`, `"fal-ai"`, `"featherless-ai"`, `"fireworks-ai"`, `"groq"`, `"hf-inference"`, `"novita"`, `"nscale"`, `"openai"`, `"ovhcloud"`, `"publicai"`, `"replicate"`, `"scaleway"`, `"together"`, `"wavespeed"` or `"zai-org"`.
            Defaults to "auto": automatic routing, which defaults to "fastest" provider; you can
            switch to "cheapest" or "preferred" provider order at https://hf.co/settings/inference-providers.
            If model is a URL or `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.
            Note: for better compatibility with OpenAI's client, `token` has been aliased as `api_key`. Those 2
            arguments are mutually exclusive and have the exact same behavior.
        timeout (`float`, `optional`):
            The maximum number of seconds to wait for a response from the server. Defaults to None, meaning it will loop until the server is available.
        headers (`dict[str, str]`, `optional`):
            Additional headers to send to the server. By default only the authorization and user-agent headers are sent.
            Values in this dictionary will override the default values.
        bill_to (`str`, `optional`):
            The billing account to use for the requests. By default the requests are billed on the user's account.
            Requests can only be billed to an organization the user is a member of, and which has subscribed to Enterprise Hub.
        cookies (`dict[str, str]`, `optional`):
            Additional cookies to send to the server.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        api_key (`str`, `optional`):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
    """

    provider: PROVIDER_OR_POLICY_T | None

    @validate_hf_hub_args
    def __init__(
        self,
        model: str | None = None,
        *,
        provider: PROVIDER_OR_POLICY_T | None = None,
        token: str | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        bill_to: str | None = None,
        # OpenAI compatibility
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if model is not None and base_url is not None:
            raise ValueError(
                "Received both `model` and `base_url` arguments. Please provide only one of them."
                " `base_url` is an alias for `model` to make the API compatible with OpenAI's client."
                " If using `base_url` for chat completion, the `/chat/completions` suffix path will be appended to the base url."
                " When passing a URL as `model`, the client will not append any suffix path to it."
            )
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        if isinstance(token, bool):
            raise TypeError("`token` must be a string or `None`.")
        self.model: str | None = base_url or model
        self.token: str | None = token

        self.headers = {**headers} if headers is not None else {}
        if bill_to is not None:
            if (
                constants.HUGGINGFACE_HEADER_X_BILL_TO in self.headers
                and self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO] != bill_to
            ):
                warnings.warn(
                    f"Overriding existing '{self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO]}' value in headers with '{bill_to}'.",
                    UserWarning,
                )
            self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO] = bill_to

            if token is not None and not token.startswith("hf_"):
                warnings.warn(
                    "You've provided an external provider's API key, so requests will be billed directly by the provider. "
                    "The `bill_to` parameter is only applicable for Hugging Face billing and will be ignored.",
                    UserWarning,
                )

        # Configure provider
        self.provider = provider  # type: ignore[assignment]

        self.cookies = cookies
        self.timeout = timeout

        self.exit_stack = ExitStack()

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.close()

    def close(self):
        self.exit_stack.close()

    @overload
    def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[False] = ...
    ) -> bytes: ...

    @overload
    def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[True] = ...
    ) -> Iterable[str]: ...

    @overload
    def _inner_post(self, request_parameters: RequestParameters, *, stream: bool = False) -> bytes | Iterable[str]: ...

    def _inner_post(self, request_parameters: RequestParameters, *, stream: bool = False) -> bytes | Iterable[str]:
        """Make a request to the inference server."""
        # TODO: this should be handled in provider helpers directly
        if request_parameters.task in TASKS_EXPECTING_IMAGES and "Accept" not in request_parameters.headers:
            request_parameters.headers["Accept"] = "image/png"

        try:
            response = self.exit_stack.enter_context(
                get_session().stream(
                    "POST",
                    request_parameters.url,
                    json=request_parameters.json,
                    content=request_parameters.data,
                    headers=request_parameters.headers,
                    cookies=self.cookies,
                    timeout=self.timeout,
                )
            )
            hf_raise_for_status(response)
            if stream:
                return response.iter_lines()
            else:
                return response.read()
        except TimeoutError as error:
            # Convert any `TimeoutError` to a `InferenceTimeoutError`
            raise InferenceTimeoutError(f"Inference call timed out: {request_parameters.url}") from error  # type: ignore
        except HfHubHTTPError as error:
            if error.response.status_code == 422 and request_parameters.task != "unknown":
                msg = str(error.args[0])
                if len(error.response.text) > 0:
                    msg += f"{os.linesep}{error.response.text}{os.linesep}"
                error.args = (msg,) + error.args[1:]
            if error.response.status_code == 504 and not stream and request_parameters.task == "conversational":
                msg = str(error.args[0])
                msg += (
                    f"{os.linesep}Note: the request timed out before the model finished generating."
                    " If you are generating long outputs (e.g. long reasoning traces), pass `stream=True`"
                    " to receive tokens as they are generated and avoid hitting this timeout."
                )
                error.args = (msg,) + error.args[1:]
            raise

    def audio_classification(
        self,
        audio: ContentT,
        *,
        model: str | None = None,
        top_k: int | None = None,
        function_to_apply: Optional["AudioClassificationOutputTransform"] = None,
    ) -> list[AudioClassificationOutputElement]:
        """
        Perform audio classification on the provided audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content to classify. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model to use for audio classification. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio classification will be used.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"AudioClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `list[AudioClassificationOutputElement]`: List of [`AudioClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.audio_classification("audio.flac")
        [
            AudioClassificationOutputElement(score=0.4976358711719513, label='hap'),
            AudioClassificationOutputElement(score=0.3677836060523987, label='neu'),
            ...
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="audio-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={"function_to_apply": function_to_apply, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return AudioClassificationOutputElement.parse_obj_as_list(response)

    def audio_to_audio(
        self,
        audio: ContentT,
        *,
        model: str | None = None,
    ) -> list[AudioToAudioOutputElement]:
        """
        Performs multiple tasks related to audio-to-audio depending on the model (eg: speech enhancement, source separation).

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content for the model. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model can be any model which takes an audio file and returns another audio file. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio_to_audio will be used.

        Returns:
            `list[AudioToAudioOutputElement]`: A list of [`AudioToAudioOutputElement`] items containing audios label, content-type, and audio content in blob.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> audio_output = client.audio_to_audio("audio.flac")
        >>> for i, item in enumerate(audio_output):
        >>>     with open(f"output_{i}.flac", "wb") as f:
                    f.write(item.blob)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="audio-to-audio", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        audio_output = AudioToAudioOutputElement.parse_obj_as_list(response)
        for item in audio_output:
            item.blob = base64.b64decode(item.blob)
        return audio_output

    def automatic_speech_recognition(
        self,
        audio: ContentT,
        *,
        model: str | None = None,
        extra_body: dict | None = None,
    ) -> AutomaticSpeechRecognitionOutput:
        """
        Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
            model (`str`, *optional*):
                The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for ASR will be used.
            extra_body (`dict`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            [`AutomaticSpeechRecognitionOutput`]: An item containing the transcribed text and optionally the timestamp chunks.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.automatic_speech_recognition("hello_world.flac").text
        "hello world"
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="automatic-speech-recognition", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={**(extra_body or {})},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_params=request_parameters)
        return AutomaticSpeechRecognitionOutput.parse_obj_as_instance(response)

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = None,
        stream: Literal[False] = False,
        frequency_penalty: float | None = None,
        logit_bias: list[float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: ChatCompletionInputGrammarType | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        stream_options: ChatCompletionInputStreamOptions | None = None,
        temperature: float | None = None,
        tool_choice: Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"] | None = None,
        tool_prompt: str | None = None,
        tools: list[ChatCompletionInputTool] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ) -> ChatCompletionOutput: ...

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = None,
        stream: Literal[True] = True,
        frequency_penalty: float | None = None,
        logit_bias: list[float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: ChatCompletionInputGrammarType | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        stream_options: ChatCompletionInputStreamOptions | None = None,
        temperature: float | None = None,
        tool_choice: Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"] | None = None,
        tool_prompt: str | None = None,
        tools: list[ChatCompletionInputTool] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ) -> Iterable[ChatCompletionStreamOutput]: ...

    @overload
    def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = None,
        stream: bool = False,
        frequency_penalty: float | None = None,
        logit_bias: list[float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: ChatCompletionInputGrammarType | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        stream_options: ChatCompletionInputStreamOptions | None = None,
        temperature: float | None = None,
        tool_choice: Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"] | None = None,
        tool_prompt: str | None = None,
        tools: list[ChatCompletionInputTool] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]: ...

    def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = None,
        stream: bool = False,
        # Parameters from ChatCompletionInput (handled manually)
        frequency_penalty: float | None = None,
        logit_bias: list[float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: ChatCompletionInputGrammarType | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        stream_options: ChatCompletionInputStreamOptions | None = None,
        temperature: float | None = None,
        tool_choice: Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"] | None = None,
        tool_prompt: str | None = None,
        tools: list[ChatCompletionInputTool] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
        """
        A method for completing conversations using a specified language model.

        > [!TIP]
        > The `client.chat_completion` method is aliased as `client.chat.completions.create` for compatibility with OpenAI's client.
        > Inputs and outputs are strictly the same and using either syntax will yield the same results.
        > Check out the [Inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility)
        > for more details about OpenAI's compatibility.

        > [!TIP]
        > You can pass provider-specific parameters to the model by using the `extra_body` argument.

        Args:
            messages (List of [`ChatCompletionInputMessage`]):
                Conversation history consisting of roles and content pairs.
            model (`str`, *optional*):
                The model to use for chat-completion. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for chat-based text-generation will be used.
                See https://huggingface.co/tasks/text-generation for more details.
                If `model` is a model ID, it is passed to the server as the `model` parameter. If you want to define a
                custom URL while setting `model` in the request payload, you must set `base_url` when initializing [`InferenceClient`].
            frequency_penalty (`float`, *optional*):
                Penalizes new tokens based on their existing frequency
                in the text so far. Range: [-2.0, 2.0]. Defaults to 0.0.
            logit_bias (`list[float]`, *optional*):
                Adjusts the likelihood of specific tokens appearing in the generated output.
            logprobs (`bool`, *optional*):
                Whether to return log probabilities of the output tokens or not. If true, returns the log
                probabilities of each output token returned in the content of message.
            max_tokens (`int`, *optional*):
                Maximum number of tokens allowed in the response. Defaults to 100.
            n (`int`, *optional*):
                The number of completions to generate for each prompt.
            presence_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
                text so far, increasing the model's likelihood to talk about new topics.
            response_format ([`ChatCompletionInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            seed (Optional[`int`], *optional*):
                Seed for reproducible control flow. Defaults to None.
            stop (`list[str]`, *optional*):
                Up to four strings which trigger the end of the response.
                Defaults to None.
            stream (`bool`, *optional*):
                Enable realtime streaming of responses. Defaults to False.
            stream_options ([`ChatCompletionInputStreamOptions`], *optional*):
                Options for streaming completions.
            temperature (`float`, *optional*):
                Controls randomness of the generations. Lower values ensure
                less random completions. Range: [0, 2]. Defaults to 1.0.
            top_logprobs (`int`, *optional*):
                An integer between 0 and 5 specifying the number of most likely tokens to return at each token
                position, each with an associated log probability. logprobs must be set to true if this parameter is
                used.
            top_p (`float`, *optional*):
                Fraction of the most likely next words to sample from.
                Must be between 0 and 1. Defaults to 1.0.
            tool_choice ([`ChatCompletionInputToolChoiceClass`] or [`ChatCompletionInputToolChoiceEnum`], *optional*):
                The tool to use for the completion. Defaults to "auto".
            tool_prompt (`str`, *optional*):
                A prompt to be appended before the tools.
            tools (List of [`ChatCompletionInputTool`], *optional*):
                A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
                provide a list of functions the model may generate JSON inputs for.
            extra_body (`dict`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            [`ChatCompletionOutput`] or Iterable of [`ChatCompletionStreamOutput`]:
            Generated text returned from the server:
            - if `stream=False`, the generated text is returned as a [`ChatCompletionOutput`] (default).
            - if `stream=True`, the generated text is returned token by token as a sequence of [`ChatCompletionStreamOutput`].

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:

        ```py
        >>> from huggingface_hub import InferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> client.chat_completion(messages, max_tokens=100)
        ChatCompletionOutput(
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason='eos_token',
                    index=0,
                    message=ChatCompletionOutputMessage(
                        role='assistant',
                        content='The capital of France is Paris.',
                        name=None,
                        tool_calls=None
                    ),
                    logprobs=None
                )
            ],
            created=1719907176,
            id='',
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            object='text_completion',
            system_fingerprint='2.0.4-sha-f426a33',
            usage=ChatCompletionOutputUsage(
                completion_tokens=8,
                prompt_tokens=17,
                total_tokens=25
            )
        )
        ```

        Example using streaming:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> for token in client.chat_completion(messages, max_tokens=10, stream=True):
        ...     print(token)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        (...)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ```

        Example using OpenAI's syntax:
        ```py
        # instead of `from openai import OpenAI`
        from huggingface_hub import InferenceClient

        # instead of `client = OpenAI(...)`
        client = InferenceClient(
            base_url=...,
            api_key=...,
        )

        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=True,
            max_tokens=1024,
        )

        for chunk in output:
            print(chunk.choices[0].delta.content)
        ```

        Example using a third-party provider directly with extra (provider-specific) parameters. Usage will be billed on your Together AI account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="together",  # Use Together AI provider
        ...     api_key="<together_api_key>",  # Pass your Together API key directly
        ... )
        >>> client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     messages=[{"role": "user", "content": "What is the capital of France?"}],
        ...     extra_body={"safety_model": "Meta-Llama/Llama-Guard-7b"},
        ... )
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="novita",  # Use Novita provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     messages=[{"role": "user", "content": "What is the capital of France?"}],
        ... )
        ```

        Example using Image + Text as input:
        ```py
        >>> from huggingface_hub import InferenceClient

        # provide a remote URL
        >>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        # or a base64-encoded image
        >>> image_path = "/path/to/image.jpeg"
        >>> with open(image_path, "rb") as f:
        ...     base64_image = base64.b64encode(f.read()).decode("utf-8")
        >>> image_url = f"data:image/jpeg;base64,{base64_image}"

        >>> client = InferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
        >>> output = client.chat.completions.create(
        ...     messages=[
        ...         {
        ...             "role": "user",
        ...             "content": [
        ...                 {
        ...                     "type": "image_url",
        ...                     "image_url": {"url": image_url},
        ...                 },
        ...                 {
        ...                     "type": "text",
        ...                     "text": "Describe this image in one sentence.",
        ...                 },
        ...             ],
        ...         },
        ...     ],
        ... )
        >>> output
        The image depicts the iconic Statue of Liberty situated in New York Harbor, New York, on a clear day.
        ```

        Example using tools:
        ```py
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": "What's the weather like the next 3 days in San Francisco, CA?",
        ...     },
        ... ]
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_current_weather",
        ...             "description": "Get the current weather",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format"],
        ...             },
        ...         },
        ...     },
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_n_day_weather_forecast",
        ...             "description": "Get an N-day weather forecast",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                     "num_days": {
        ...                         "type": "integer",
        ...                         "description": "The number of days to forecast",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format", "num_days"],
        ...             },
        ...         },
        ...     },
        ... ]

        >>> response = client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-70B-Instruct",
        ...     messages=messages,
        ...     tools=tools,
        ...     tool_choice="auto",
        ...     max_tokens=500,
        ... )
        >>> response.choices[0].message.tool_calls[0].function
        ChatCompletionOutputFunctionDefinition(
            arguments={
                'location': 'San Francisco, CA',
                'format': 'fahrenheit',
                'num_days': 3
            },
            name='get_n_day_weather_forecast',
            description=None
        )
        ```

        Example using response_format:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I see and when?",
        ...     },
        ... ]
        >>> response_format = {
        ...     "type": "json",
        ...     "value": {
        ...         "properties": {
        ...             "location": {"type": "string"},
        ...             "activity": {"type": "string"},
        ...             "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...             "animals": {"type": "array", "items": {"type": "string"}},
        ...         },
        ...         "required": ["location", "activity", "animals_seen", "animals"],
        ...     },
        ... }
        >>> response = client.chat_completion(
        ...     messages=messages,
        ...     response_format=response_format,
        ...     max_tokens=500,
        ... )
        >>> response.choices[0].message.content
        '{\n\n"activity": "bike ride",\n"animals": ["puppy", "cat", "raccoon"],\n"animals_seen": 3,\n"location": "park"}'
        ```
        """
        # Since `chat_completion(..., model=xxx)` is also a payload parameter for the server, we need to handle 'model' differently.
        # `self.model` takes precedence over 'model' argument for building URL.
        # `model` takes precedence for payload value.
        model_id_or_url = self.model or model
        payload_model = model or self.model

        # Get the provider helper
        provider_helper = get_provider_helper(
            self.provider,
            task="conversational",
            model=model_id_or_url
            if model_id_or_url is not None and model_id_or_url.startswith(("http://", "https://"))
            else payload_model,
        )

        # Prepare the payload
        parameters = {
            "model": payload_model,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tool_prompt": tool_prompt,
            "tools": tools,
            "top_logprobs": top_logprobs,
            "top_p": top_p,
            "stream": stream,
            "stream_options": stream_options,
            **(extra_body or {}),
        }
        request_parameters = provider_helper.prepare_request(
            inputs=messages,
            parameters=parameters,
            headers=self.headers,
            model=model_id_or_url,
            api_key=self.token,
        )
        data = self._inner_post(request_parameters, stream=stream)

        if stream:
            return _stream_chat_completion_response(data)  # type: ignore

        return ChatCompletionOutput.parse_obj_as_instance(data)  # type: ignore

    def document_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: str | None = None,
        doc_stride: int | None = None,
        handle_impossible_answer: bool | None = None,
        lang: str | None = None,
        max_answer_len: int | None = None,
        max_question_len: int | None = None,
        max_seq_len: int | None = None,
        top_k: int | None = None,
        word_boxes: list[list[float] | str] | None = None,
    ) -> list[DocumentQuestionAnsweringOutputElement]:
        """
        Answer questions on document images.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for the context. It can be raw bytes, an image file, or a URL to an online image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the document question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended document question answering model will be used.
                Defaults to None.
            doc_stride (`int`, *optional*):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using doc_stride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Can return less than top_k
                answers if there are not enough options available within the context.
            word_boxes (`list[Union[list[float], str`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If provided, the inference will skip the OCR
                step and use the provided bounding boxes instead.
        Returns:
            `list[DocumentQuestionAnsweringOutputElement]`: a list of [`DocumentQuestionAnsweringOutputElement`] items containing the predicted label, associated probability, word ids, and page number.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.


        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
        [DocumentQuestionAnsweringOutputElement(answer='us-001', end=16, score=0.9999666213989258, start=16)]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="document-question-answering", model=model_id)
        inputs: dict[str, Any] = {"question": question, "image": _b64_encode(image)}
        request_parameters = provider_helper.prepare_request(
            inputs=inputs,
            parameters={
                "doc_stride": doc_stride,
                "handle_impossible_answer": handle_impossible_answer,
                "lang": lang,
                "max_answer_len": max_answer_len,
                "max_question_len": max_question_len,
                "max_seq_len": max_seq_len,
                "top_k": top_k,
                "word_boxes": word_boxes,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return DocumentQuestionAnsweringOutputElement.parse_obj_as_list(response)

    def feature_extraction(
        self,
        text: str | list[str],
        *,
        normalize: bool | None = None,
        prompt_name: str | None = None,
        truncate: bool | None = None,
        truncation_direction: Literal["left", "right"] | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        model: str | None = None,
    ) -> "np.ndarray":
        """
        Generate embeddings for a given text or batch of texts.

        Args:
            text (`str` or `list[str]`):
                The text or list of texts to embed.
            model (`str`, *optional*):
                The model to use for the feature extraction task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended feature extraction model will be used.
                Defaults to None.
            normalize (`bool`, *optional*):
                Whether to normalize the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            prompt_name (`str`, *optional*):
                The name of the prompt that should be used by for encoding. If not set, no prompt will be applied.
                Must be a key in the `Sentence Transformers` configuration `prompts` dictionary.
                For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ",...},
                then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the prompt text will be prepended before any text to encode.
            truncate (`bool`, *optional*):
                Whether to truncate the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            truncation_direction (`Literal["left", "right"]`, *optional*):
                Which side of the input should be truncated when `truncate=True` is passed.
            dimensions (`int`, *optional*):
                The number of dimensions the resulting output embeddings should have.
                Only available on OpenAI-compatible embedding endpoints.
            encoding_format (`Literal["float", "base64"]`, *optional*):
                The format of the output embeddings. Either "float" or "base64".
                Only available on OpenAI-compatible embedding endpoints.

        Returns:
            `np.ndarray`: The embedding representing the input text(s) as a float32 numpy array.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.feature_extraction("Hi, who are you?")
        array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
        [-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
        ...,
        [ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="feature-extraction", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(provider_helper.get_response(response), dtype="float32")

    def fill_mask(
        self,
        text: str,
        *,
        model: str | None = None,
        targets: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[FillMaskOutputElement]:
        """
        Fill in a hole with a missing word (token to be precise).

        Args:
            text (`str`):
                a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask).
            model (`str`, *optional*):
                The model to use for the fill mask task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended fill mask model will be used.
            targets (`list[str`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocabulary. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.
        Returns:
            `list[FillMaskOutputElement]`: a list of [`FillMaskOutputElement`] items containing the predicted label, associated
            probability, token reference, and completed text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.fill_mask("The goal of life is <mask>.")
        [
            FillMaskOutputElement(score=0.06897063553333282, token=11098, token_str=' happiness', sequence='The goal of life is happiness.'),
            FillMaskOutputElement(score=0.06554922461509705, token=45075, token_str=' immortality', sequence='The goal of life is immortality.')
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="fill-mask", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={"targets": targets, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return FillMaskOutputElement.parse_obj_as_list(response)

    def image_classification(
        self,
        image: ContentT,
        *,
        model: str | None = None,
        function_to_apply: Optional["ImageClassificationOutputTransform"] = None,
        top_k: int | None = None,
    ) -> list[ImageClassificationOutputElement]:
        """
        Perform image classification on the given image using the specified model.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to classify. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for image classification. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image classification will be used.
            function_to_apply (`"ImageClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
        Returns:
            `list[ImageClassificationOutputElement]`: a list of [`ImageClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        [ImageClassificationOutputElement(label='Blenheim spaniel', score=0.9779096841812134), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"function_to_apply": function_to_apply, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return ImageClassificationOutputElement.parse_obj_as_list(response)

    def image_segmentation(
        self,
        image: ContentT,
        *,
        model: str | None = None,
        mask_threshold: float | None = None,
        overlap_mask_area_threshold: float | None = None,
        subtask: Optional["ImageSegmentationSubtask"] = None,
        threshold: float | None = None,
    ) -> list[ImageSegmentationOutputElement]:
        """
        Perform image segmentation on the given image using the specified model.

        > [!WARNING]
        > You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to segment. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.
            mask_threshold (`float`, *optional*):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*):
                Mask overlap threshold to eliminate small, disconnected segments.
            subtask (`"ImageSegmentationSubtask"`, *optional*):
                Segmentation task to be performed, depending on model capabilities.
            threshold (`float`, *optional*):
                Probability threshold to filter out predicted masks.
        Returns:
            `list[ImageSegmentationOutputElement]`: A list of [`ImageSegmentationOutputElement`] items containing the segmented masks and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_segmentation("cat.jpg")
        [ImageSegmentationOutputElement(score=0.989008, label='LABEL_184', mask=<PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-segmentation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "mask_threshold": mask_threshold,
                "overlap_mask_area_threshold": overlap_mask_area_threshold,
                "subtask": subtask,
                "threshold": threshold,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        output = ImageSegmentationOutputElement.parse_obj_as_list(response)
        for item in output:
            item.mask = _b64_to_image(item.mask)  # type: ignore
        return output

    def image_to_image(
        self,
        image: ContentT,
        prompt: str | None = None,
        *,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        model: str | None = None,
        target_size: ImageToImageTargetSize | None = None,
        **kwargs,
    ) -> "Image":
        """
        Perform image-to-image translation using a specified model.

        > [!WARNING]
        > You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image for translation. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            prompt (`str`, *optional*):
                The text prompt to guide the image generation.
            negative_prompt (`str`, *optional*):
                One prompt to guide what NOT to include in image generation.
            num_inference_steps (`int`, *optional*):
                For diffusion models. The number of denoising steps. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*):
                For diffusion models. A higher guidance scale value encourages the model to generate images closely
                linked to the text prompt at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            target_size (`ImageToImageTargetSize`, *optional*):
                The size in pixels of the output image. This parameter is only supported by some providers and for
                specific models. It will be ignored when unsupported.

        Returns:
            `Image`: The translated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
        >>> image.save("tiger.jpg")
        ```

        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-to-image", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "target_size": target_size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                **kwargs,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return _bytes_to_image(response)

    def image_to_video(
        self,
        image: ContentT,
        *,
        model: str | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        num_frames: float | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        target_size: ImageToVideoTargetSize | None = None,
        **kwargs,
    ) -> bytes:
        """
        Generate a video from an input image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image to generate a video from. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            prompt (`str`, *optional*):
                The text prompt to guide the video generation.
            negative_prompt (`str`, *optional*):
                One prompt to guide what NOT to include in video generation.
            num_frames (`float`, *optional*):
                The num_frames parameter determines how many video frames are generated.
            num_inference_steps (`int`, *optional*):
                For diffusion models. The number of denoising steps. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*):
                For diffusion models. A higher guidance scale value encourages the model to generate videos closely
                linked to the text prompt at the expense of lower image quality.
            seed (`int`, *optional*):
                The seed to use for the video generation.
            target_size (`ImageToVideoTargetSize`, *optional*):
                The size in pixel of the output video frames.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            seed (`int`, *optional*):
                Seed for the random number generator.

        Returns:
            `bytes`: The generated video.

        Examples:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> video = client.image_to_video("cat.jpg", model="Wan-AI/Wan2.2-I2V-A14B", prompt="turn the cat into a tiger")
        >>> with open("tiger.mp4", "wb") as f:
        ...     f.write(video)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-to-video", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "target_size": target_size,
                **kwargs,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return response

    def image_to_text(self, image: ContentT, *, model: str | None = None) -> ImageToTextOutput:
        """
        Takes an input image and return text.

        Models can have very different outputs depending on your use case (image captioning, optical character recognition
        (OCR), Pix2Struct, etc.). Please have a look to the model card to learn more about a model's specificities.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image to caption. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            [`ImageToTextOutput`]: The generated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_to_text("cat.jpg")
        'a cat standing in a grassy field '
        >>> client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        'a dog laying on the grass next to a flower pot '
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-to-text", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        output_list: list[ImageToTextOutput] = ImageToTextOutput.parse_obj_as_list(response)
        return output_list[0]

    def object_detection(
        self, image: ContentT, *, model: str | None = None, threshold: float | None = None
    ) -> list[ObjectDetectionOutputElement]:
        """
        Perform object detection on the given image using the specified model.

        > [!WARNING]
        > You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to detect objects on. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for object detection. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for object detection (DETR) will be used.
            threshold (`float`, *optional*):
                The probability necessary to make a prediction.
        Returns:
            `list[ObjectDetectionOutputElement]`: A list of [`ObjectDetectionOutputElement`] items containing the bounding boxes and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If the request output is not a List.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.object_detection("people.jpg")
        [ObjectDetectionOutputElement(score=0.9486683011054993, label='person', box=ObjectDetectionBoundingBox(xmin=59, ymin=39, xmax=420, ymax=510)), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="object-detection", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"threshold": threshold},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return ObjectDetectionOutputElement.parse_obj_as_list(response)

    def question_answering(
        self,
        question: str,
        context: str,
        *,
        model: str | None = None,
        align_to_words: bool | None = None,
        doc_stride: int | None = None,
        handle_impossible_answer: bool | None = None,
        max_answer_len: int | None = None,
        max_question_len: int | None = None,
        max_seq_len: int | None = None,
        top_k: int | None = None,
    ) -> QuestionAnsweringOutputElement | list[QuestionAnsweringOutputElement]:
        """
        Retrieve the answer to a question from a given text.

        Args:
            question (`str`):
                Question to be answered.
            context (`str`):
                The context of the question.
            model (`str`):
                The model to use for the question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint.
            align_to_words (`bool`, *optional*):
                Attempts to align the answer to real words. Improves quality on space separated languages. Might hurt
                on non-space-separated languages (like Japanese or Chinese)
            doc_stride (`int`, *optional*):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using docStride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.

        Returns:
            Union[`QuestionAnsweringOutputElement`, list[`QuestionAnsweringOutputElement`]]:
                When top_k is 1 or not provided, it returns a single `QuestionAnsweringOutputElement`.
                When top_k is greater than 1, it returns a list of `QuestionAnsweringOutputElement`.
        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
        QuestionAnsweringOutputElement(answer='Clara', end=16, score=0.9326565265655518, start=11)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"question": question, "context": context},
            parameters={
                "align_to_words": align_to_words,
                "doc_stride": doc_stride,
                "handle_impossible_answer": handle_impossible_answer,
                "max_answer_len": max_answer_len,
                "max_question_len": max_question_len,
                "max_seq_len": max_seq_len,
                "top_k": top_k,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        # Parse the response as a single `QuestionAnsweringOutputElement` when top_k is 1 or not provided, or a list of `QuestionAnsweringOutputElement` to ensure backward compatibility.
        output = QuestionAnsweringOutputElement.parse_obj(response)
        return output

    def sentence_similarity(
        self, sentence: str, other_sentences: list[str], *, model: str | None = None
    ) -> list[float]:
        """
        Compute the semantic similarity between a sentence and a list of other sentences by comparing their embeddings.

        Args:
            sentence (`str`):
                The main sentence to compare to others.
            other_sentences (`list[str]`):
                The list of sentences to compare to.
            model (`str`, *optional*):
                The model to use for the sentence similarity task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended sentence similarity model will be used.
                Defaults to None.

        Returns:
            `list[float]`: The embedding representing the input text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.sentence_similarity(
        ...     "Machine learning is so easy.",
        ...     other_sentences=[
        ...         "Deep learning is so straightforward.",
        ...         "This is so difficult, like rocket science.",
        ...         "I can't believe how much I struggled with this.",
        ...     ],
        ... )
        [0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="sentence-similarity", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"source_sentence": sentence, "sentences": other_sentences},
            parameters={},
            extra_payload={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def summarization(
        self,
        text: str,
        *,
        model: str | None = None,
        clean_up_tokenization_spaces: bool | None = None,
        generate_parameters: dict[str, Any] | None = None,
        truncation: Optional["SummarizationTruncationStrategy"] = None,
    ) -> SummarizationOutput:
        """
        Generate a summary of a given text using a specified model.

        Args:
            text (`str`):
                The input text to summarize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for summarization will be used.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            generate_parameters (`dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.
            truncation (`"SummarizationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
        Returns:
            [`SummarizationOutput`]: The generated summary text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.summarization("The Eiffel tower...")
        SummarizationOutput(generated_text="The Eiffel tower is one of the most famous landmarks in the world....")
        ```
        """
        parameters = {
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "generate_parameters": generate_parameters,
            "truncation": truncation,
        }
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="summarization", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters=parameters,
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return SummarizationOutput.parse_obj_as_list(response)[0]

    def table_question_answering(
        self,
        table: dict[str, Any],
        query: str,
        *,
        model: str | None = None,
        padding: Optional["Padding"] = None,
        sequential: bool | None = None,
        truncation: bool | None = None,
    ) -> TableQuestionAnsweringOutputElement:
        """
        Retrieve the answer to a question from information given in a table.

        Args:
            table (`str`):
                A table of data represented as a dict of lists where entries are headers and the lists are all the
                values, all lists must have the same size.
            query (`str`):
                The query in plain text that you want to ask the table.
            model (`str`):
                The model to use for the table-question-answering task. Can be a model ID hosted on the Hugging Face
                Hub or a URL to a deployed Inference Endpoint.
            padding (`"Padding"`, *optional*):
                Activates and controls padding.
            sequential (`bool`, *optional*):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            truncation (`bool`, *optional*):
                Activates and controls truncation.

        Returns:
            [`TableQuestionAnsweringOutputElement`]: a table question answering output containing the answer, coordinates, cells and the aggregator used.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> query = "How many stars does the transformers repository have?"
        >>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
        >>> client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
        TableQuestionAnsweringOutputElement(answer='36542', coordinates=[[0, 1]], cells=['36542'], aggregator='AVERAGE')
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="table-question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"query": query, "table": table},
            parameters={"model": model, "padding": padding, "sequential": sequential, "truncation": truncation},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TableQuestionAnsweringOutputElement.parse_obj_as_instance(response)

    def tabular_classification(self, table: dict[str, Any], *, model: str | None = None) -> list[str]:
        """
        Classifying a target category (a group) based on a set of attributes.

        Args:
            table (`dict[str, Any]`):
                Set of attributes to classify.
            model (`str`, *optional*):
                The model to use for the tabular classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular classification model will be used.
                Defaults to None.

        Returns:
            `List`: a list of labels, one per row in the initial table.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> table = {
        ...     "fixed_acidity": ["7.4", "7.8", "10.3"],
        ...     "volatile_acidity": ["0.7", "0.88", "0.32"],
        ...     "citric_acid": ["0", "0", "0.45"],
        ...     "residual_sugar": ["1.9", "2.6", "6.4"],
        ...     "chlorides": ["0.076", "0.098", "0.073"],
        ...     "free_sulfur_dioxide": ["11", "25", "5"],
        ...     "total_sulfur_dioxide": ["34", "67", "13"],
        ...     "density": ["0.9978", "0.9968", "0.9976"],
        ...     "pH": ["3.51", "3.2", "3.23"],
        ...     "sulphates": ["0.56", "0.68", "0.82"],
        ...     "alcohol": ["9.4", "9.8", "12.6"],
        ... }
        >>> client.tabular_classification(table=table, model="julien-c/wine-quality")
        ["5", "5", "5"]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="tabular-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=None,
            extra_payload={"table": table},
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def tabular_regression(self, table: dict[str, Any], *, model: str | None = None) -> list[float]:
        """
        Predicting a numerical target value given a set of attributes/features in a table.

        Args:
            table (`dict[str, Any]`):
                Set of attributes stored in a table. The attributes used to predict the target can be both numerical and categorical.
            model (`str`, *optional*):
                The model to use for the tabular regression task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular regression model will be used.
                Defaults to None.

        Returns:
            `List`: a list of predicted numerical target values.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> table = {
        ...     "Height": ["11.52", "12.48", "12.3778"],
        ...     "Length1": ["23.2", "24", "23.9"],
        ...     "Length2": ["25.4", "26.3", "26.5"],
        ...     "Length3": ["30", "31.2", "31.1"],
        ...     "Species": ["Bream", "Bream", "Bream"],
        ...     "Width": ["4.02", "4.3056", "4.6961"],
        ... }
        >>> client.tabular_regression(table, model="scikit-learn/Fish-Weight")
        [110, 120, 130]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="tabular-regression", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=None,
            parameters={},
            extra_payload={"table": table},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def text_classification(
        self,
        text: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        function_to_apply: Optional["TextClassificationOutputTransform"] = None,
    ) -> list[TextClassificationOutputElement]:
        """
        Perform text classification (e.g. sentiment-analysis) on the given text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the text classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended text classification model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"TextClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `list[TextClassificationOutputElement]`: a list of [`TextClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.text_classification("I like you")
        [
            TextClassificationOutputElement(label='POSITIVE', score=0.9998695850372314),
            TextClassificationOutputElement(label='NEGATIVE', score=0.0001304351753788069),
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "function_to_apply": function_to_apply,
                "top_k": top_k,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TextClassificationOutputElement.parse_obj_as_list(response)[0]  # type: ignore

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Literal[True],
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> Iterable[TextGenerationStreamOutput]: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Literal[False] | None = None,
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> TextGenerationOutput: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[False] | None = None,
        stream: Literal[True],
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,  # Manual default value
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> Iterable[str]: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[False] | None = None,
        stream: Literal[False] | None = None,
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> str: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: bool | None = None,
        stream: bool | None = None,
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> str | TextGenerationOutput | Iterable[str] | Iterable[TextGenerationStreamOutput]: ...

    def text_generation(
        self,
        prompt: str,
        *,
        details: bool | None = None,
        stream: bool | None = None,
        model: str | None = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: str | None = None,
        best_of: int | None = None,
        decoder_input_details: bool | None = None,
        do_sample: bool | None = None,
        frequency_penalty: float | None = None,
        grammar: TextGenerationInputGrammarType | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_n_tokens: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool | None = None,
    ) -> str | TextGenerationOutput | Iterable[str] | Iterable[TextGenerationStreamOutput]:
        """
        Given a prompt, generate the following text.

        > [!TIP]
        > If you want to generate a response from chat messages, you should use the [`InferenceClient.chat_completion`] method.
        > It accepts a list of messages instead of a single text prompt and handles the chat templating for you.

        Args:
            prompt (`str`):
                Input text.
            details (`bool`, *optional*):
                By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
                probabilities, seed, finish reason, etc.). Only available for models running on with the
                `text-generation-inference` backend.
            stream (`bool`, *optional*):
                By default, text_generation returns the full generated text. Pass `stream=True` if you want a stream of
                tokens to be returned. Only available for models running on with the `text-generation-inference`
                backend.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            adapter_id (`str`, *optional*):
                Lora adapter id.
            best_of (`int`, *optional*):
                Generate best_of sequences and return the one if the highest token logprobs.
            decoder_input_details (`bool`, *optional*):
                Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
                into account. Defaults to `False`.
            do_sample (`bool`, *optional*):
                Activate logits sampling
            frequency_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
                the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            grammar ([`TextGenerationInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            max_new_tokens (`int`, *optional*):
                Maximum number of generated tokens. Defaults to 100.
            repetition_penalty (`float`, *optional*):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`, *optional*):
                Whether to prepend the prompt to the generated text
            seed (`int`, *optional*):
                Random sampling seed
            stop (`list[str]`, *optional*):
                Stop generating tokens if a member of `stop` is generated.
            temperature (`float`, *optional*):
                The value used to module the logits distribution.
            top_n_tokens (`int`, *optional*):
                Return information about the `top_n_tokens` most likely tokens at each generation step, instead of
                just the sampled token.
            top_k (`int`, *optional`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`, *optional`):
                Truncate inputs tokens to the given size.
            typical_p (`float`, *optional`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`, *optional*):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)

        Returns:
            `Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `Iterable[str]`
            - if `stream=False` and `details=True`, the generated text is returned with more details as a [`~huggingface_hub.TextGenerationOutput`]
            - if `details=True` and `stream=True`, the generated text is returned token by token as a iterable of [`~huggingface_hub.TextGenerationStreamOutput`]

        Raises:
            `ValidationError`:
                If input values are not valid. No HTTP call is made to the server.
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        # Case 1: generate text
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful for large generation.
        >>> for token in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
        ...     print(token)
        100
        %
        open
        source
        and
        built
        to
        be
        easy
        to
        use
        .

        # Case 3: get more details about the generation process.
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
        TextGenerationOutput(
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationDetails(
                finish_reason='length',
                generated_tokens=12,
                seed=None,
                prefill=[
                    TextGenerationPrefillOutputToken(id=487, text='The', logprob=None),
                    TextGenerationPrefillOutputToken(id=53789, text=' hugging', logprob=-13.171875),
                    (...)
                    TextGenerationPrefillOutputToken(id=204, text=' ', logprob=-7.0390625)
                ],
                tokens=[
                    TokenElement(id=1425, text='100', logprob=-1.0175781, special=False),
                    TokenElement(id=16, text='%', logprob=-0.0463562, special=False),
                    (...)
                    TokenElement(id=25, text='.', logprob=-0.5703125, special=False)
                ],
                best_of_sequences=None
            )
        )

        # Case 4: iterate over the generated tokens with more details.
        # Last object is more complete, containing the full generated text and the finish reason.
        >>> for details in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
        ...     print(details)
        ...
        TextGenerationStreamOutput(token=TokenElement(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(
            id=25,
            text='.',
            logprob=-0.5703125,
            special=False),
