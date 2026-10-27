        raise click.ClickException(f"Shell '{shell}' is not supported for completion.")
    return completion_cls(app, {}, "hf", _COMPLETE_VAR).source()


def _show_completion(value: bool) -> None:
    if not value:
        return
    click.echo(_completion_script(_detect_shell()))
    raise click.exceptions.Exit()


def _install_completion(value: bool) -> None:
    if not value:
        return
    shell = _detect_shell()
    if shell == "fish":
        path = _FISH_COMPLETION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_completion_script("fish"))
    elif shell in _RC_ACTIVATION:
        path, activation = _RC_ACTIVATION[shell]
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text() if path.exists() else ""
        if activation not in existing:
            with path.open("a") as file:
                file.write(f"\n{activation}\n")
    else:
        raise click.ClickException(f"Shell '{shell}' is not supported for completion.")
    click.echo(f"{shell} completion installed in {path}. Restart your shell for it to take effect.")
    raise click.exceptions.Exit()


InstallCompletionOpt = Annotated[
    bool,
    Option(
        "--install-completion",
        callback=_install_completion,
        is_eager=True,
        help="Install completion for the current shell.",
    ),
]

ShowCompletionOpt = Annotated[
    bool,
    Option(
        "--show-completion",
        callback=_show_completion,
        is_eager=True,
        help="Show completion for the current shell, to copy it or customize the installation.",
    ),
]
<|endoftext|>"""
Implementation of a custom transfer agent for the transfer type "multipart" for
git-lfs.

Inspired by:
github.com/cbartz/git-lfs-swift-transfer-agent/blob/master/git_lfs_swift_transfer.py

Spec is: github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md


To launch debugger while developing:

``` [lfs "customtransfer.multipart"]
path = /path/to/huggingface_hub/.venv/bin/python args = -m debugpy --listen 5678
--wait-for-client
/path/to/huggingface_hub/src/huggingface_hub/commands/huggingface_cli.py
lfs-multipart-upload ```"""

import json
import os
import subprocess
import sys
from typing import Annotated

from huggingface_hub.errors import CLIError
from huggingface_hub.lfs import LFS_MULTIPART_UPLOAD_COMMAND

from ..utils import get_session, hf_raise_for_status, logging
from ..utils._lfs import SliceFileObj
from ._framework import Argument
from ._output import out


logger = logging.get_logger(__name__)


def lfs_enable_largefiles(
    path: Annotated[
        str,
        Argument(
            help="Local path to repository you want to configure.",
        ),
    ],
) -> None:
    """
    Configure your repository to enable upload of files > 5GB.

    This command sets up git-lfs to use the custom multipart transfer agent
    which enables efficient uploading of large files in chunks.
    """
    local_path = os.path.abspath(path)
    if not os.path.isdir(local_path):
        raise CLIError("This does not look like a valid git repo.")
    subprocess.run(
        "git config lfs.customtransfer.multipart.path hf".split(),
        check=True,
        cwd=local_path,
    )
    subprocess.run(
        f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(),
        check=True,
        cwd=local_path,
    )
    out.result("Local repo set up for largefiles", path=local_path)


def write_msg(msg: dict):
    """Write out the message in Line delimited JSON."""
    msg_str = json.dumps(msg) + "\n"
    sys.stdout.write(msg_str)
    sys.stdout.flush()


def read_msg() -> dict | None:
    """Read Line delimited JSON from stdin."""
    msg = json.loads(sys.stdin.readline().strip())

    if "terminate" in (msg.get("type"), msg.get("event")):
        # terminate message received
        return None

    if msg.get("event") not in ("download", "upload"):
        logger.critical("Received unexpected message")
        sys.exit(1)

    return msg


def lfs_multipart_upload() -> None:
    """Internal git-lfs custom transfer agent for multipart uploads.

    This function implements the custom transfer protocol for git-lfs multipart uploads.
    Handles chunked uploads of large files to Hugging Face Hub.
    """
    # Immediately after invoking a custom transfer process, git-lfs
    # sends initiation data to the process over stdin.
    # This tells the process useful information about the configuration.
    init_msg = json.loads(sys.stdin.readline().strip())
    if not (init_msg.get("event") == "init" and init_msg.get("operation") == "upload"):
        write_msg({"error": {"code": 32, "message": "Wrong lfs init operation"}})
        sys.exit(1)

    # The transfer process should use the information it needs from the
    # initiation structure, and also perform any one-off setup tasks it
    # needs to do. It should then respond on stdout with a simple empty
    # confirmation structure, as follows:
    write_msg({})

    # After the initiation exchange, git-lfs will send any number of
    # transfer requests to the stdin of the transfer process, in a serial sequence.
    while True:
        msg = read_msg()
        if msg is None:
            # When all transfers have been processed, git-lfs will send
            # a terminate event to the stdin of the transfer process.
            # On receiving this message the transfer process should
            # clean up and terminate. No response is expected.
            sys.exit(0)

        oid = msg["oid"]
        filepath = msg["path"]
        completion_url = msg["action"]["href"]
        header = msg["action"]["header"]
        chunk_size = int(header.pop("chunk_size"))
        presigned_urls: list[str] = list(header.values())

        # Send a "started" progress event to allow other workers to start.
        # Otherwise they're delayed until first "progress" event is reported,
        # i.e. after the first 5GB by default (!)
        write_msg(
            {
                "event": "progress",
                "oid": oid,
                "bytesSoFar": 1,
                "bytesSinceLast": 0,
            }
        )

        parts = []
        with open(filepath, "rb") as file:
            for i, presigned_url in enumerate(presigned_urls):
                with SliceFileObj(
                    file,
                    seek_from=i * chunk_size,
                    read_limit=chunk_size,
                ) as data:
                    r = get_session().put(presigned_url, data=data)
                    hf_raise_for_status(r)
                    parts.append(
                        {
                            "etag": r.headers.get("etag"),
                            "partNumber": i + 1,
                        }
                    )
                    # In order to support progress reporting while data is uploading / downloading,
                    # the transfer process should post messages to stdout
                    write_msg(
                        {
                            "event": "progress",
                            "oid": oid,
                            "bytesSoFar": (i + 1) * chunk_size,
                            "bytesSinceLast": chunk_size,
                        }
                    )

        r = get_session().post(
            completion_url,
            json={
                "oid": oid,
                "parts": parts,
            },
        )
        hf_raise_for_status(r)

        write_msg({"event": "complete", "oid": oid})
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
"""CLI error handling utilities."""

import traceback
from collections.abc import Callable

from huggingface_hub.errors import (
    BucketNotFoundError,
    CLIError,
    CLIExtensionInstallError,
    DeviceCodeError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HfUriError,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    LocalTokenNotFoundError,
    OfflineModeIsEnabled,
    OIDCError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def _format_repo_not_found(error: RepositoryNotFoundError) -> str:
    label = error.repo_type.capitalize() if error.repo_type else "Repository"
    if error.repo_id:
        msg = f"{label} '{error.repo_id}' not found."
    else:
        msg = f"{label} not found."
    msg += "\nIf the repo is private, make sure you are authenticated and your token has the required permissions."

    msg += "\nIf the repo does not exist, create it with: "
    if error.repo_id is not None:
        type_flag = f" --type {error.repo_type}" if error.repo_type and error.repo_type != "model" else ""
        msg += f"hf repos create {error.repo_id}{type_flag}"
    else:
        msg += "hf repos create <repo_id>"

    return msg


def _format_gated_repo(error: GatedRepoError) -> str:
    label = error.repo_type if error.repo_type else "repository"
    if error.repo_id:
        return f"Access denied. {label.capitalize()} '{error.repo_id}' requires approval."
    return f"Access denied. This {label} requires approval."


def _format_bucket_not_found(error: BucketNotFoundError) -> str:
    if error.bucket_id:
        msg = f"Bucket '{error.bucket_id}' not found."
        cmd = f"hf buckets create {error.bucket_id}"
    else:
        msg = "Bucket not found."
        cmd = "hf buckets create <bucket_id>"
    msg += "\nIf the bucket is private, make sure you are authenticated and your token has the required permissions."
    msg += f"\nIf the bucket does not exist, create it with: {cmd}"
    return msg


def _format_entry_not_found(error: RemoteEntryNotFoundError) -> str:
    label = error.repo_type if error.repo_type else "repository"
    url = str(error.response.url) if error.response else None
    if error.repo_id:
        msg = f"File not found in {label} '{error.repo_id}'."
    else:
        msg = f"File not found in {label}."
    if url:
        msg += f"\nURL: {url}"
    return msg


def _format_local_entry_not_found(error: LocalEntryNotFoundError) -> str:
    cause = error.__cause__
    if cause is not None:
        return f"Local entry not found. {cause}"
    return f"Local entry not found. {error}"


def _format_incomplete_snapshot(error: IncompleteSnapshotError) -> str:
    msg = _format_local_entry_not_found(error)
    msg += f"\nIncomplete snapshot available at: {error.snapshot_path}"
    return msg


def _format_revision_not_found(error: RevisionNotFoundError) -> str:
    label = error.repo_type if error.repo_type else "repository"
    if error.repo_id:
        return f"Revision not found in {label} '{error.repo_id}'."
    return f"Revision not found in {label}. Check the revision parameter."


def _format_cli_error(error: CLIError) -> str:
    """No traceback, just the error message."""
    return str(error)


def _format_cli_extension_install_error(error: CLIExtensionInstallError) -> str:
    """Format a CLI extension installation error.

    The error is likely to be a tricky subprocess error to investigate. In this specific case we want to format the
    traceback of the root cause while keeping the "nicely formatted" error message of the CLIExtensionInstallError
    as a 1-line message.
    """
    cause_tb = (
        "".join(traceback.format_exception(type(error.__cause__), error.__cause__, error.__cause__.__traceback__))
        if error.__cause__ is not None
        else ""
    )
    return f"{cause_tb}\n{error}"


CLI_ERROR_MAPPINGS: dict[type[Exception], Callable[..., str]] = {
    OfflineModeIsEnabled: lambda error: str(error),
    # GatedRepoError must come before RepositoryNotFoundError (it's a subclass).
    GatedRepoError: _format_gated_repo,
    BucketNotFoundError: _format_bucket_not_found,
    RepositoryNotFoundError: _format_repo_not_found,
    RevisionNotFoundError: _format_revision_not_found,
    LocalTokenNotFoundError: lambda _: "Not logged in. Run 'hf auth login' first.",
    OIDCError: lambda error: f"OIDC Exchange failed. {error}",
    DeviceCodeError: lambda error: f"Login failed: {error}",
    RemoteEntryNotFoundError: _format_entry_not_found,
    # IncompleteSnapshotError must come before LocalEntryNotFoundError (it's a subclass).
    IncompleteSnapshotError: _format_incomplete_snapshot,
    LocalEntryNotFoundError: _format_local_entry_not_found,
    EntryNotFoundError: lambda error: str(error),
    HfHubHTTPError: lambda error: str(error),
    HfUriError: lambda error: f"Invalid HF URI: {error.uri}. {error.msg}",
    ValueError: lambda error: f"Invalid value. {error}",
    CLIExtensionInstallError: _format_cli_extension_install_error,
    CLIError: _format_cli_error,
}


def format_known_exception(error: Exception) -> str | None:
    for exc_type, formatter in CLI_ERROR_MAPPINGS.items():
        if isinstance(error, exc_type):
            return formatter(error)
    return None
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
"""Pretty ANSI help formatter for the `hf` CLI."""

from collections.abc import Iterable

import click

from huggingface_hub.utils import ANSI


class StyledHelpFormatter(click.HelpFormatter):
    def write_heading(self, heading: str) -> None:
        styled = ANSI.underline(heading + ":")
        self.write(f"{'':>{self.current_indent}}{styled}\n")

    def write_dl(self, rows: Iterable[tuple[str, str]], col_max: int = 30, col_spacing: int = 2) -> None:
        rows = [(ANSI.bold(first), second) for first, second in rows]
        super().write_dl(rows, col_max=col_max, col_spacing=col_spacing)


class StyledContext(click.Context):
    formatter_class = StyledHelpFormatter
<|endoftext|>"""CLI commands for Hugging Face Inference Endpoints."""

import shlex
from typing import Annotated

import click

from huggingface_hub._inference_endpoints import (
    InferenceEndpointHardware,
    InferenceEndpointScalingMetric,
    InferenceEndpointType,
    _set_parallelism_in_image,
)
from huggingface_hub.errors import CLIError, HfHubHTTPError

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    RevisionOpt,
    SearchOpt,
    SecretsFileOpt,
    SecretsOpt,
    SoftChoice,
    TokenOpt,
    get_hf_api,
    parse_env_map,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


ie_cli = typer_factory(help="Manage Hugging Face Inference Endpoints.")

catalog_app = typer_factory(help="Interact with the Inference Endpoints catalog.")


NameArg = Annotated[
    str,
    Argument(help="Endpoint name."),
]
NameOpt = Annotated[
    str | None,
    Option(help="Endpoint name."),
]

NamespaceOpt = Annotated[
    str | None,
    Option(
        help="The namespace associated with the Inference Endpoint. Defaults to the current user's namespace.",
    ),
]

# Maps the CLI `--engine` values to the `model.image` variant keys of the API payload. `huggingface` and
# `huggingfaceNeuron` are left out on purpose: they are the managed Hugging Face images and take no image URL.
ENGINE_IMAGE_KEYS = {
    "custom": "custom",
    "hf-serve": "hfServe",
    "llamacpp": "llamacpp",
    "sglang": "sGLang",
    "tei": "tei",
    "tgi": "tgi",
    "tgi-neuron": "tgiNeuron",
    "vllm": "vLLM",
    "vllm-neuron": "vLLMNeuron",
}

EngineOpt = Annotated[
    str | None,
    Option(
        "--engine",
        click_type=SoftChoice(list(ENGINE_IMAGE_KEYS)),
        help="Managed engine image to run --custom-image with (e.g. 'vllm'). Defaults to an arbitrary container.",
    ),
]

HealthRouteOpt = Annotated[
    str | None,
    Option(
        help="Health check route exposed by the container (e.g. '/health'). Requires --custom-image.",
    ),
]

PortOpt = Annotated[
    int | None,
    Option(
        help="Port the container listens on (e.g. 30000). Requires --custom-image.",
    ),
]

TensorParallelSizeOpt = Annotated[
    int | None,
    Option(
        "--tensor-parallel-size",
        help="Number of accelerators to shard a single model copy across (vLLM and SGLang engines only).",
    ),
]

DataParallelSizeOpt = Annotated[
    int | None,
    Option(
        "--data-parallel-size",
        help="Number of model copies to run, one per accelerator (vLLM engine only).",
    ),
]


@ie_cli.command("list | ls", examples=["hf endpoints ls", "hf endpoints ls --namespace my-org"])
def ls(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Lists all Inference Endpoints for the given namespace."""
    api = get_hf_api(token=token)
    try:
        endpoints = api.list_inference_endpoints(namespace=namespace, token=token)
    except HfHubHTTPError as error:
        out.error(f"Listing failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    results = []
    for endpoint in endpoints:
        raw = endpoint.raw
        status = raw.get("status", {})
        model = raw.get("model", {})
        compute = raw.get("compute", {})
        provider = raw.get("provider", {})
        results.append(
            {
                "name": raw.get("name", ""),
                "model": model.get("repository", "") if isinstance(model, dict) else "",
                "status": status.get("state", "") if isinstance(status, dict) else "",
                "task": model.get("task", "") if isinstance(model, dict) else "",
                "framework": model.get("framework", "") if isinstance(model, dict) else "",
                "instance": compute.get("instanceType", "") if isinstance(compute, dict) else "",
                "vendor": provider.get("vendor", "") if isinstance(provider, dict) else "",
                "region": provider.get("region", "") if isinstance(provider, dict) else "",
            }
        )
    out.table(results, id_key="name")


# A denylist like the Endpoints UI uses, so a new server-side status shows up instead of silently disappearing.
_UNDEPLOYABLE_STATUSES = {"deprecated", "not_available"}


def _is_deployable(hw: InferenceEndpointHardware) -> bool:
    """Whether the namespace can deploy one replica on this hardware right now: usable status and enough quota."""
    return (
        hw.status not in _UNDEPLOYABLE_STATUSES and hw.max_accelerators - hw.used_accelerators >= hw.num_accelerators
    )


@ie_cli.command(
    "hardware",
    examples=[
        "hf endpoints hardware",
        "hf endpoints hardware --vendor aws --accelerator gpu",
    ],
)
def hardware(
    namespace: NamespaceOpt = None,
    vendor: Annotated[
        str | None,
        Option(help="Only show hardware hosted by this cloud provider (e.g. 'aws')."),
    ] = None,
    region: Annotated[
        str | None,
        Option(help="Only show hardware available in this cloud region (e.g. 'us-east-1')."),
    ] = None,
    accelerator: Annotated[
        str | None,
        Option(help="Only show hardware with this accelerator (e.g. 'cpu', 'gpu', 'neuron')."),
    ] = None,
    instance_type: Annotated[
        str | None,
        Option(help="Only show hardware of this instance type (e.g. 'nvidia-l4')."),
    ] = None,
    show_all: Annotated[
        bool,
        Option(
            "-a",
            "--all",
            help="Also show hardware that cannot be deployed on right now (unavailable, deprecated or out of quota).",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """List the hardware available to deploy an Inference Endpoint on.

    Only the hardware the namespace can deploy on right now is listed: a usable status, and enough accelerator
    quota left for one replica. Use `--all` to list every combination the API returns.

    Quota is per namespace, so pass the same `--namespace` you will pass to `hf endpoints deploy`. Prices are in
    USD, per replica per hour.
    """
    api = get_hf_api(token=token)
    hardware_list = api.list_inference_endpoints_hardware(namespace=namespace, token=token)

    # Both sides lowercased: relying on the server's casing would turn a change into a silently empty result.
    matching = [
        hw
        for hw in hardware_list
        if (vendor is None or hw.vendor.lower() == vendor.lower())
        and (region is None or hw.region.lower() == region.lower())
        and (accelerator is None or hw.accelerator.lower() == accelerator.lower())
        and (instance_type is None or hw.instance_type.lower() == instance_type.lower())
    ]
    visible = [hw for hw in matching if show_all or _is_deployable(hw)]
    # Smallest size first per instance type; 'instance_size' can't be the key, 'x16' sorts before 'x2'.
    visible.sort(key=lambda hw: (hw.vendor, hw.region, hw.accelerator, hw.instance_type, hw.num_accelerators))

    # Add a quota column, keeping the rest of the dict whole since '--format json' emits it.
    items = [_dataclass_to_dict(hw) | {"quota": f"{hw.used_accelerators}/{hw.max_accelerators}"} for hw in visible]
    out.table(
        items,
        # Redundant and always-null columns are dropped here but stay in the items for '--format json'.
        headers=[
            "vendor",
            "region",
            "accelerator",
            "instance_type",
            "instance_size",
            "memory_gb",
            "gpu_memory_gb",
            "price_per_hour",
            "quota",
            "status",
        ],
        id_key="id",
    )
    # The example must be one 'deploy' accepts, and with '--all' the first row may not be.
    if (hw := next((hw for hw in visible if _is_deployable(hw)), None)) is not None:
        out.hint(
            f"Deploy on one of these, e.g.: hf endpoints deploy my-endpoint --repo <repo> --framework <framework> "
            f"--vendor {hw.vendor} --region {hw.region} --accelerator {hw.accelerator} "
            f"--instance-type {hw.instance_type} --instance-size {hw.instance_size}"
        )
    elif visible:  # only reachable with '--all'
        out.hint("None of these can be deployed on right now, see the QUOTA and STATUS columns.")
    elif matching:  # all matches were dropped as undeployable
        out.hint("Use '--all' to also show hardware that cannot be deployed on right now.")
    else:
        # Nothing matched at all: a filter value is probably a typo, so name the ones that exist nowhere.
        unknown = [
            f"{flag} '{value}' (valid: {', '.join(sorted(valid))})"
            for flag, value, valid in (
                ("--vendor", vendor, {hw.vendor for hw in hardware_list}),
                ("--region", region, {hw.region for hw in hardware_list}),
                ("--accelerator", accelerator, {hw.accelerator for hw in hardware_list}),
                ("--instance-type", instance_type, {hw.instance_type for hw in hardware_list}),
            )
            if value is not None and value.lower() not in {v.lower() for v in valid}
        ]
        out.hint(
            f"No such hardware: {'; '.join(unknown)}."
            if unknown
            # Each value exists, they just never occur together.
            else "No hardware matches all of these filters at once. Try dropping one."
        )


@ie_cli.command(
    name="deploy",
    examples=[
        "hf endpoints deploy my-endpoint --repo gpt2 --framework pytorch ...",
        "hf endpoints deploy my-endpoint --repo openai/gpt-oss-120b --framework custom --engine vllm "
        "--custom-image vllm/vllm-openai:v0.23.0 --tensor-parallel-size 8 ...",
    ],
)
def deploy(
    name: NameArg,
    repo: Annotated[
        str,
        Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b').",
        ),
    ],
    framework: Annotated[
        str,
        Option(
            help="The machine learning framework used for the model (e.g. 'vllm').",
        ),
    ],
    accelerator: Annotated[
        str,
        Option(
            help="The hardware accelerator to be used for inference (e.g. 'cpu').",
        ),
    ],
    instance_size: Annotated[
        str,
        Option(
            help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
        ),
    ],
    instance_type: Annotated[
        str,
        Option(
            help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
        ),
    ],
    region: Annotated[
        str,
        Option(
            help="The cloud region in which the Inference Endpoint will be created (e.g. 'us-east-1').",
        ),
    ],
    vendor: Annotated[
        str,
        Option(
            help="The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. 'aws').",
        ),
    ],
    *,
    namespace: NamespaceOpt = None,
    task: Annotated[
        str | None,
        Option(
            help="The task on which to deploy the model (e.g. 'text-classification').",
        ),
    ] = None,
    token: TokenOpt = None,
    min_replica: Annotated[
        int,
        Option(
            help="The minimum number of replicas (instances) to keep running for the Inference Endpoint.",
        ),
    ] = 1,
    max_replica: Annotated[
        int,
        Option(
            help="The maximum number of replicas (instances) to scale to for the Inference Endpoint.",
        ),
    ] = 1,
    scale_to_zero_timeout: Annotated[
        int | None,
        Option(
            help="The duration in minutes before an inactive endpoint is scaled to zero.",
        ),
    ] = None,
    scaling_metric: Annotated[
        InferenceEndpointScalingMetric | None,
        Option(
            help="The metric reference for scaling.",
        ),
    ] = None,
    scaling_threshold: Annotated[
        float | None,
        Option(
            help="The scaling metric threshold used to trigger a scale up. Ignored when scaling metric is not provided.",
        ),
    ] = None,
    revision: RevisionOpt = None,
    custom_image: Annotated[
        str | None,
        Option(
            "--custom-image",
            help="Docker image URL for the container to run (e.g. 'nexagi/sglang:v0.5.12'). Requires '--framework custom'.",
        ),
    ] = None,
    engine: EngineOpt = None,
    health_route: HealthRouteOpt = None,
    port: PortOpt = None,
    container_registry_username: Annotated[
        str | None,
        Option(
            help="Username used to authenticate with the registry hosting --custom-image.",
        ),
    ] = None,
    container_registry_password: Annotated[
        str | None,
        Option(
            help="Optional password used to authenticate with the registry hosting --custom-image.",
        ),
    ] = None,
    tensor_parallel_size: TensorParallelSizeOpt = None,
    data_parallel_size: DataParallelSizeOpt = None,
    container_command: Annotated[
        str | None,
        Option(
            "--container-command",
            help=(
                "Override the container entrypoint, as a quoted string split into tokens "
                '(e.g. "python -m sglang.launch_server").'
            ),
        ),
    ] = None,
    container_args: Annotated[
        str | None,
        Option(
            "--container-args",
            help=(
                "Arguments appended to the container entrypoint, as a quoted string split into tokens "
                '(e.g. "--tp 8 --reasoning-parser qwen3").'
            ),
        ),
    ] = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    endpoint_type: Annotated[
        str | None,
        Option(
            "--type",
            click_type=SoftChoice(InferenceEndpointType),
            help="Endpoint access type. Defaults to 'authenticated' (token-gated, publicly reachable).",
        ),
    ] = None,
) -> None:
    """Deploy an Inference Endpoint from a Hub repository.

    Run `hf endpoints hardware` to list the valid `--vendor`, `--region`, `--accelerator`, `--instance-type` and
    `--instance-size` combinations.
    """
    custom_image_dict = _build_custom_image(
        custom_image,
        engine=engine,
        health_route=health_route,
        port=port,
        container_registry_username=container_registry_username,
        container_registry_password=container_registry_password,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    env_map = {key: value or "" for key, value in parse_env_map(env, env_file).items()}
    secrets_map = {key: value or "" for key, value in parse_env_map(secrets, secrets_file).items()}

    # Only forward the values the user actually set and let `create_inference_endpoint` own the defaults.
    params: dict = {}
    if endpoint_type is not None:
        params["type"] = endpoint_type
    if custom_image_dict is not None:
        params["custom_image"] = custom_image_dict
    if container_registry_username is not None:
        params["container_registry_username"] = container_registry_username
    if container_registry_password is not None:
        params["container_registry_password"] = container_registry_password
    if container_command:
        params["container_command"] = shlex.split(container_command)
    if container_args:
        params["container_args"] = shlex.split(container_args)
    if env_map:
        params["env"] = env_map
    if secrets_map:
        params["secrets"] = secrets_map

    api = get_hf_api(token=token)
    endpoint = api.create_inference_endpoint(
        name=name,
        repository=repo,
        framework=framework,
        accelerator=accelerator,
        instance_size=instance_size,
        instance_type=instance_type,
        region=region,
        vendor=vendor,
        namespace=namespace,
        task=task,
        token=token,
        min_replica=min_replica,
        max_replica=max_replica,
        scaling_metric=scaling_metric,
        scaling_threshold=scaling_threshold,
        scale_to_zero_timeout=scale_to_zero_timeout,
        revision=revision,
        **params,
    )
    out.dict(endpoint.raw)
    out.hint(f"Use 'hf endpoints describe {name}' to check the deployment status.")


@catalog_app.command(
    name="deploy",
    examples=[
        "hf endpoints catalog deploy --repo meta-llama/Llama-3.2-1B-Instruct",
        "hf endpoints catalog deploy --recipe sizzling-biryani-g4xsi1ac",
    ],
)
def deploy_from_catalog(
    repo: Annotated[
        str | None,
        Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b')."
            " Deploys its default recipe. Mutually exclusive with --recipe.",
        ),
    ] = None,
    recipe: Annotated[
        str | None,
        Option(
            help="The id of the catalog recipe to deploy, as listed by 'hf endpoints catalog ls'."
            " Mutually exclusive with --repo.",
        ),
    ] = None,
    name: NameOpt = None,
    accelerator: Annotated[
        str | None,
        Option(
            click_type=SoftChoice(["cpu", "gpu", "neuron"]),
            help="The hardware accelerator to be used for inference. Only with --repo.",
        ),
    ] = None,
    gguf_file: Annotated[
        str | None,
        Option(
            help="The GGUF file to deploy, for models that have one recipe per quant. Only with --repo.",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Deploy an Inference Endpoint from the Model Catalog.

    Catalog models are deployed through a recipe: a hardware and engine combination that has been tested for them.
    Pass --repo to deploy the default recipe of a model, optionally narrowed down with --accelerator and
    --gguf-file, or pass --recipe to deploy an exact recipe listed by `hf endpoints catalog ls`.
    """
    # `--repo` used to be required, so a bare `catalog deploy` used to fail in click. Keep the message flag-shaped.
    if (repo is None) == (recipe is None):
        raise CLIError("Provide exactly one of --repo or --recipe. List recipes with 'hf endpoints catalog ls'.")

    api = get_hf_api(token=token)
    try:
        endpoint = api.create_inference_endpoint_from_catalog(
            repo_id=repo,
            recipe_id=recipe,
            name=name,
            accelerator=accelerator,
            gguf_file=gguf_file,
            namespace=namespace,
            token=token,
        )
    except HfHubHTTPError as error:
        out.error(f"Deployment failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    out.dict(endpoint.raw)
    out.hint(f"Use 'hf endpoints describe {endpoint.name}' to check the deployment status.")


def list_catalog(
    accelerator: Annotated[
        str | None,
        Option(
            click_type=SoftChoice(["cpu", "gpu", "neuron"]),
            help="Only show recipes running on this accelerator.",
        ),
    ] = None,
    engine: Annotated[
        str | None,
        Option(
            click_type=SoftChoice(["llamacpp", "sglang", "tei", "vllm"]),
            help="Only show recipes running this inference engine.",
        ),
    ] = None,
    license: Annotated[
        str | None,
        Option(help="Only show models under this license (e.g. 'Apache 2.0')."),
    ] = None,
    task: Annotated[
        str | None,
        Option(help="Only show models for this task (e.g. 'text-generation')."),
    ] = None,
    search: SearchOpt = None,
    limit: Annotated[int | None, Option(help="Limit the number of models to return.")] = None,
    token: TokenOpt = None,
) -> None:
    """List the models available in the Model Catalog.

    One row per recipe, i.e. per tested way of deploying a model. Deploy one with
    `hf endpoints catalog deploy --recipe <RECIPE_ID>`.
    """
    api = get_hf_api(token=token)
    try:
        models = api.list_inference_catalog(
            accelerator=accelerator,
            engine=engine,
            license=license,
            task=task,
            search=search,
            limit=limit,
            token=token,
        )
    except HfHubHTTPError as error:
        out.error(f"Catalog fetch failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    items = [
        {
            "repo_id": model.repo_id,
            "task": model.task,
            "license": model.license,
            "accelerator": recipe.accelerator,
            "engine": recipe.engine,
            "gguf_file": recipe.gguf_file,
            "recipe_id": recipe.id,
        }
        for model in models
        for recipe in model.recipes
    ]
    out.table(items, id_key="recipe_id")
    if items:
        out.hint(f"Deploy one with: hf endpoints catalog deploy --recipe {items[0]['recipe_id']}")


catalog_app.command(name="list | ls", examples=["hf endpoints catalog ls", "hf endpoints catalog ls --engine vllm"])(
    list_catalog
)
ie_cli.command(name="list-catalog", hidden=True)(list_catalog)


ie_cli.add_group(catalog_app, name="catalog")


@ie_cli.command(examples=["hf endpoints describe my-endpoint"])
def describe(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get information about an existing endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.get_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        out.error(f"Fetch failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    out.dict(endpoint.raw)


@ie_cli.command(
    examples=[
        "hf endpoints update my-endpoint --min-replica 2",
        "hf endpoints update my-endpoint --tensor-parallel-size 8",
        'hf endpoints update my-endpoint --container-args "--enable-auto-tool-choice --tool-call-parser lfm2"',
    ]
)
def update(
    name: NameArg,
    namespace: NamespaceOpt = None,
    repo: Annotated[
        str | None,
        Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b').",
        ),
    ] = None,
    accelerator: Annotated[
        str | None,
        Option(
            help="The hardware accelerator to be used for inference (e.g. 'cpu').",
        ),
    ] = None,
    instance_size: Annotated[
        str | None,
        Option(
            help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
        ),
    ] = None,
    instance_type: Annotated[
        str | None,
        Option(
            help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
        ),
    ] = None,
    framework: Annotated[
        str | None,
        Option(
            help="The machine learning framework used for the model (e.g. 'custom').",
        ),
    ] = None,
    revision: Annotated[
        str | None,
        Option(
            help="The specific model revision to deploy on the Inference Endpoint (e.g. '6c0e6080953db56375760c0471a8c5f2929baf11').",
        ),
    ] = None,
    task: Annotated[
        str | None,
        Option(
            help="The task on which to deploy the model (e.g. 'text-classification').",
        ),
    ] = None,
    custom_image: Annotated[
        str | None,
        Option(
            "--custom-image",
            help=(
                "Docker image URL for the container to run (e.g. 'nexagi/sglang:v0.5.12'). Replaces the image "
                "currently configured on the endpoint rather than patching it, so pass the engine and container "
                "settings you want to keep along with it, run 'hf endpoints describe NAME' first to see them."
            ),
        ),
    ] = None,
    engine: EngineOpt = None,
    health_route: HealthRouteOpt = None,
    port: PortOpt = None,
    tensor_parallel_size: TensorParallelSizeOpt = None,
    data_parallel_size: DataParallelSizeOpt = None,
    container_command: Annotated[
        str | None,
        Option(
            "--container-command",
            help=(
                "Override the container entrypoint, as a quoted string split into tokens "
                '(e.g. "python -m sglang.launch_server"). Replaces the current value; '
                "pass an empty string to clear it."
            ),
        ),
    ] = None,
    container_args: Annotated[
        str | None,
        Option(
            "--container-args",
            help=(
                "Arguments appended to the container entrypoint, as a quoted string split into tokens "
                '(e.g. "--enable-auto-tool-choice --tool-call-parser lfm2"). Replaces the arguments currently '
                "set on the endpoint rather than adding to them, so include the ones you want to keep, run "
                "'hf endpoints describe NAME' first to see them. Pass an empty string to clear them."
            ),
        ),
    ] = None,
    min_replica: Annotated[
        int | None,
        Option(
            help="The minimum number of replicas (instances) to keep running for the Inference Endpoint.",
        ),
    ] = None,
    max_replica: Annotated[
        int | None,
        Option(
            help="The maximum number of replicas (instances) to scale to for the Inference Endpoint.",
        ),
    ] = None,
    scale_to_zero_timeout: Annotated[
        int | None,
        Option(
            help="The duration in minutes before an inactive endpoint is scaled to zero.",
        ),
    ] = None,
    scaling_metric: Annotated[
        InferenceEndpointScalingMetric | None,
        Option(
            help="The metric reference for scaling.",
        ),
    ] = None,
    scaling_threshold: Annotated[
        float | None,
        Option(
            help="The scaling metric threshold used to trigger a scale up. Ignored when scaling metric is not provided.",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update an existing endpoint."""
    # Exactly one path carries the sizes: into the image built here when `--custom-image` gives us one, otherwise
    # as kwargs, for `update_inference_endpoint` to merge into the image the endpoint currently runs.
    custom_image_dict = _build_custom_image(
        custom_image,
        engine=engine,
        health_route=health_route,
        port=port,
        tensor_parallel_size=tensor_parallel_size if custom_image is not None else None,
        data_parallel_size=data_parallel_size if custom_image is not None else None,
    )

    api = get_hf_api(token=token)
    try:
        endpoint = api.update_inference_endpoint(
            name=name,
            namespace=namespace,
            repository=repo,
            framework=framework,
            revision=revision,
            task=task,
            custom_image=custom_image_dict,
            container_command=shlex.split(container_command) if container_command is not None else None,
            container_args=shlex.split(container_args) if container_args is not None else None,
            tensor_parallel_size=tensor_parallel_size if custom_image is None else None,
            data_parallel_size=data_parallel_size if custom_image is None else None,
            accelerator=accelerator,
            instance_size=instance_size,
            instance_type=instance_type,
            min_replica=min_replica,
            max_replica=max_replica,
            scale_to_zero_timeout=scale_to_zero_timeout,
            scaling_metric=scaling_metric,
            scaling_threshold=scaling_threshold,
            token=token,
        )
    except HfHubHTTPError as error:
        out.error(f"Update failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error
    out.dict(endpoint.raw)


@ie_cli.command(examples=["hf endpoints delete my-endpoint"])
def delete(
    name: NameArg,
    namespace: NamespaceOpt = None,
    yes: Annotated[
        bool,
        Option("--yes", help="Skip confirmation prompts."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete an Inference Endpoint permanently."""
    out.confirm(f"Delete endpoint '{name}'?", yes=yes)

    api = get_hf_api(token=token)
    try:
        api.delete_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        out.error(f"Delete failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    out.result(f"Deleted '{name}'.", name=name)


@ie_cli.command(examples=["hf endpoints pause my-endpoint"])
def pause(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Pause an Inference Endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.pause_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        out.error(f"Pause failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    out.dict(endpoint.raw)


@ie_cli.command(examples=["hf endpoints resume my-endpoint"])
def resume(
    name: NameArg,
    namespace: NamespaceOpt = None,
    fail_if_already_running: Annotated[
        bool,
        Option(
            "--fail-if-already-running",
            help="If `True`, the method will raise an error if the Inference Endpoint is already running.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Resume an Inference Endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.resume_inference_endpoint(
            name=name,
            namespace=namespace,
            token=token,
            running_ok=not fail_if_already_running,
        )
    except HfHubHTTPError as error:
        out.error(f"Resume failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error
    out.dict(endpoint.raw)


@ie_cli.command(examples=["hf endpoints scale-to-zero my-endpoint"])
def scale_to_zero(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Scale an Inference Endpoint to zero."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.scale_to_zero_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        out.error(f"Scale To Zero failed: {error}")
        raise click.exceptions.Exit(code=error.response.status_code) from error

    out.dict(endpoint.raw)


def _build_custom_image(
    custom_image: str | None,
    *,
    engine: str | None = None,
    health_route: str | None = None,
    port: int | None = None,
    container_registry_username: str | None = None,
    container_registry_password: str | None = None,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
) -> dict | None:
    """Build the `custom_image` argument of `HfApi` from the flat image flags.

    A bare image URL describes a custom container and is passed flat, `HfApi` keys it as `custom`. `--engine`
    selects a managed engine image instead, whose config is keyed by the API's variant name. An engine this
    version doesn't know about is forwarded as typed, so the API names the bad variant.
    """
    if custom_image is None:
        # These flags all describe the container image, and the only image these commands can build is the one
        # --custom-image names. Container command/args are top-level model fields, hence not listed here.
        image_flags = {
            "--engine": engine,
            "--health-route": health_route,
            "--port": port,
            "--container-registry-username": container_registry_username,
            "--container-registry-password": container_registry_password,
            "--tensor-parallel-size": tensor_parallel_size,
            "--data-parallel-size": data_parallel_size,
        }
        if used := [flag for flag, value in image_flags.items() if value is not None]:
            raise CLIError(f"--custom-image is required when using {', '.join(used)}.")
        return None

    if container_registry_password is not None and container_registry_username is None:
        raise CLIError("--container-registry-password requires --container-registry-username.")
    if engine is not None and container_registry_username is not None:
        raise CLIError("Container registry credentials can only be set for a custom container without --engine.")

    if engine is None and (tensor_parallel_size is not None or data_parallel_size is not None):
        # Without an engine the config is a plain custom container, whose parallelism fields the API ignores
        # rather than rejects. Fail here instead of deploying something that quietly runs on one accelerator.
        raise CLIError("--tensor-parallel-size and --data-parallel-size require --engine (e.g. --engine vllm).")

    config: dict = {"url": custom_image}
    if health_route is not None:
        config["healthRoute"] = health_route
    if port is not None:
        config["port"] = port
    if engine is None:
        return config
    # Delegate the sizes so this and `update_inference_endpoint` share one rule, including the warning when
    # the engine does not declare the field.
    return _set_parallelism_in_image(
        {ENGINE_IMAGE_KEYS.get(engine, engine): config},
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )
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
"""Contains commands to interact with models on the Hugging Face Hub."""

import enum
from typing import Annotated, cast, get_args

import click

from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelSort_T
from huggingface_hub.inference._providers import PROVIDER_T
from huggingface_hub.repocard import ModelCard

from ._cli_utils import (
    REPO_LIST_DEFAULT_LIMIT,
    AuthorOpt,
    FilterOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    SoftChoice,
    TokenOpt,
    get_hf_api,
    make_expand_properties_parser,
    typer_factory,
)
from ._file_listing import list_repo_files_cmd
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


_EXPAND_PROPERTIES = sorted(get_args(ExpandModelProperty_T))
_SORT_OPTIONS = get_args(ModelSort_T)
ModelSortEnum = enum.Enum("ModelSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]
InferenceProviderEnum = enum.Enum(  # type: ignore[misc]
    "InferenceProviderEnum", {p: p for p in sorted(get_args(PROVIDER_T))}, type=str
)


ExpandOpt = Annotated[
    str | None,
    Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


models_cli = typer_factory(help="Interact with models on the Hub.")


@models_cli.command(
    "list | ls",
    examples=[
        "hf models ls --sort downloads --limit 10",
        'hf models ls --search "llama" --author meta-llama',
        "hf models ls --pipeline-tag text-generation --warm",
        "hf models ls --num-parameters min:6B,max:128B --sort likes",
        "hf models ls --no-gated --author google",
        "hf models ls --apps llama.cpp --apps vllm",
        "hf models ls --inference-provider fireworks-ai --sort downloads",
        "hf models ls --warm --search llama",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct -R",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct --tree -h",
    ],
)
def models_ls(
    repo_id: Annotated[
        str | None,
        Argument(help="Model ID (e.g. `username/repo-name`) to list files from. If omitted, lists models."),
    ] = None,
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    pipeline_tag: Annotated[
        str | None,
        Option("--pipeline-tag", help="Filter by pipeline tag (canonical task), e.g. 'summarization'."),
    ] = None,
    gated: Annotated[
        bool | None,
        Option(
            "--gated/--no-gated",
            help="Filter by gated status. '--gated' for gated only, '--no-gated' for non-gated only.",
        ),
    ] = None,
    apps: Annotated[
        list[str] | None,
        Option("--apps", help="Filter by app(s) that can run the model, e.g. 'ollama' or 'vllm'."),
    ] = None,
    num_parameters: Annotated[
        str | None,
        Option(help="Filter by parameter count, e.g. 'min:6B,max:128B'."),
    ] = None,
    inference_provider: Annotated[
        list[str] | None,
        Option(
            "--inference-provider",
            click_type=SoftChoice(InferenceProviderEnum),
            help="Filter by inference provider(s) serving the model, e.g. 'fireworks-ai'.",
        ),
    ] = None,
    warm: Annotated[
        bool,
        Option("--warm", help="Only list models currently served by at least one inference provider."),
    ] = False,
    sort: Annotated[
        ModelSortEnum | None,
        Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = REPO_LIST_DEFAULT_LIMIT,
    expand: ExpandOpt = None,
    human_readable: Annotated[
        bool,
        Option("--human-readable", "-h", help="Show sizes in human readable format (only for listing files)."),
    ] = False,
    as_tree: Annotated[
        bool,
        Option("--tree", help="List files in tree format (only for listing files)."),
    ] = False,
    recursive: Annotated[
        bool,
        Option("--recursive", "-R", help="List files recursively (only for listing files)."),
    ] = False,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
) -> None:
    """List models on the Hub, or files in a model repo.

    When called with no argument, lists models on the Hub.
    When called with a model ID, lists files in that model repo.
    """
    if repo_id is not None:
        if search is not None:
            raise click.BadParameter("Cannot use --search when listing files.")
        if author is not None:
            raise click.BadParameter("Cannot use --author when listing files.")
        if filter is not None:
            raise click.BadParameter("Cannot use --filter when listing files.")
        if pipeline_tag is not None:
            raise click.BadParameter("Cannot use --pipeline-tag when listing files.")
        if gated is not None:
            raise click.BadParameter("Cannot use --gated/--no-gated when listing files.")
        if apps is not None:
            raise click.BadParameter("Cannot use --apps when listing files.")
        if num_parameters is not None:
            raise click.BadParameter("Cannot use --num-parameters when listing files.")
        if inference_provider is not None:
            raise click.BadParameter("Cannot use --inference-provider when listing files.")
        if warm:
            raise click.BadParameter("Cannot use --warm when listing files.")
        if sort is not None:
            raise click.BadParameter("Cannot use --sort when listing files.")
        if limit != REPO_LIST_DEFAULT_LIMIT:
            raise click.BadParameter("Cannot use --limit when listing files.")
        if expand is not None:
            raise click.BadParameter("Cannot use --expand when listing files.")
        return list_repo_files_cmd(
            repo_id=repo_id,
            repo_type="model",
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            revision=revision,
            token=token,
        )

    if as_tree:
        raise click.BadParameter("Cannot use --tree when listing models.")
    if recursive:
        raise click.BadParameter("Cannot use --recursive when listing models.")
    if human_readable:
        raise click.BadParameter("Cannot use --human-readable when listing models.")
    if revision is not None:
        raise click.BadParameter("Cannot use --revision when listing models.")
    if warm and inference_provider is not None:
        raise click.BadParameter("Cannot use --warm together with --inference-provider.")
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        _dataclass_to_dict(model_info)
        for model_info in api.list_models(
            filter=filter,
            author=author,
            search=search,
            pipeline_tag=pipeline_tag,
            gated=gated,
            apps=apps,
            num_parameters=num_parameters,
            inference="warm" if warm else None,
            inference_provider=cast(list[PROVIDER_T] | None, inference_provider),
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore
        )
    ]
    out.table(results)
    if (inference_provider is not None or warm) and not expand:
        out.hint(
            "Use `--expand inferenceProviderMapping` to see which provider serves each model and the provider-specific model id."
        )


@models_cli.command(
    "info",
    examples=[
        "hf models info meta-llama/Llama-3.2-1B-Instruct",
        "hf models info Qwen/Qwen3.5-9B --expand downloads,likes,tags",
    ],
)
def models_info(
    model_id: Annotated[str, Argument(help="The model ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a model on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.model_info(repo_id=model_id, revision=revision, expand=expand)  # type: ignore
    except RepositoryNotFoundError as e:
        raise CLIError(f"Model '{model_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{model_id}'.") from e
    out.dict(info)


@models_cli.command(
    "card",
    examples=[
        "hf models card google/gemma-4-31B-it",
        "hf models card google/gemma-4-31B-it --metadata",
        "hf models card google/gemma-4-31B-it --metadata --format json",
        "hf models card google/gemma-4-31B-it --text",
    ],
)
def models_card(
    model_id: Annotated[str, Argument(help="The model ID (e.g. `username/repo-name`).")],
    metadata: Annotated[bool, Option("--metadata", help="Output only the metadata from the card.")] = False,
    text: Annotated[bool, Option("--text", help="Output only the text body (no metadata).")] = False,
    token: TokenOpt = None,
) -> None:
    """Get the model card (README) for a model on the Hub."""
    if metadata and text:
        raise CLIError("--metadata and --text are mutually exclusive.")
    card = ModelCard.load(model_id, token=token)
    if metadata:
        out.dict(card.data.to_dict())
    elif text:
        out.text(card.text)
    else:
        out.text(card.content)
        out.hint(f"Use `hf models card {model_id} --metadata` to extract only the card metadata.")
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
"""Contains commands to manage webhooks on the Hugging Face Hub."""

import enum
from typing import Annotated, get_args, get_type_hints

import click

from huggingface_hub.constants import WEBHOOK_DOMAIN_T
from huggingface_hub.hf_api import WebhookWatchedItem

from ._cli_utils import (
    SecretsFileOpt,
    SecretsOpt,
    TokenOpt,
    get_hf_api,
    parse_env_map,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import out


# Build enums dynamically from Literal types to avoid duplication
_WATCHED_TYPES = get_args(get_type_hints(WebhookWatchedItem)["type"])
WatchedItemType = enum.Enum("WatchedItemType", {t: t for t in _WATCHED_TYPES}, type=str)  # type: ignore[misc]

_DOMAIN_TYPES = get_args(WEBHOOK_DOMAIN_T)
WebhookDomain = enum.Enum("WebhookDomain", {d: d for d in _DOMAIN_TYPES}, type=str)  # type: ignore[misc]


def _parse_watch(values: list[str]) -> list[WebhookWatchedItem]:
    """Parse 'type:name' strings into WebhookWatchedItem objects.

    Args:
        values: List of strings in the format 'type:name'
            (e.g., 'model:bert-base-uncased', 'org:HuggingFace').

    Returns:
        List of WebhookWatchedItem objects.

    Raises:
        click.BadParameter: If any value doesn't match the expected format.
    """
    items = []
    valid_types = tuple(_WATCHED_TYPES)
    for v in values:
        if ":" not in v:
            raise click.BadParameter(
                f"Expected format 'type:name' (e.g. 'model:bert-base-uncased'), got '{v}'."
                f" Valid types: {', '.join(valid_types)}."
            )
        kind, name = v.split(":", 1)
        if kind not in valid_types:
            raise click.BadParameter(f"Invalid type '{kind}'. Valid types: {', '.join(valid_types)}.")
        items.append(WebhookWatchedItem(type=kind, name=name))  # type: ignore
    return items


def _parse_job_secrets(
    secrets: list[str] | None, secrets_file: str | None, job_id: str | None
) -> dict[str, str] | None:
    """Parse `--secrets` / `--secrets-file` for the Job triggered by a webhook (requires `--job-id`)."""
    secrets_map = parse_env_map(secrets, secrets_file)
    if not secrets_map:
        return None
    if job_id is None:
        raise click.BadParameter("--secrets and --secrets-file require --job-id.")
    return {key: value or "" for key, value in secrets_map.items()}


webhooks_cli = typer_factory(help="Manage webhooks on the Hub.")


@webhooks_cli.command(
    "list | ls",
    examples=[
        "hf webhooks ls",
        "hf webhooks ls --format json",
        "hf webhooks ls --format quiet",
    ],
)
def webhooks_ls(
    token: TokenOpt = None,
) -> None:
    """List all webhooks for the current user."""
    api = get_hf_api(token=token)
    results = [
        {
            "id": w.id,
            "url": w.url or "(job)",
            "disabled": w.disabled,
            "domains": w.domains or [],
            "watched": [f"{wi.type}:{wi.name}" for wi in (w.watched or [])],
        }
        for w in api.list_webhooks()
    ]
    out.table(results)


@webhooks_cli.command(
    "info",
    examples=[
        "hf webhooks info abc123",
    ],
)
def webhooks_info(
    webhook_id: Annotated[str, Argument(help="The ID of the webhook.")],
    token: TokenOpt = None,
) -> None:
    """Show full details for a single webhook."""
    api = get_hf_api(token=token)
    webhook = api.get_webhook(webhook_id)
    out.dict(webhook)


@webhooks_cli.command(
    "create",
    examples=[
        "hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased",
        "hf webhooks create --url https://example.com/hook --watch org:HuggingFace --watch model:gpt2 --domain repo",
        "hf webhooks create --job-id 687f911eaea852de79c4a50a --watch user:julien-c",
        "hf webhooks create --job-id 687f911eaea852de79c4a50a --watch bucket:my-org/my-bucket --secrets HF_TOKEN",
    ],
)
def webhooks_create(
    watch: Annotated[
        list[str],
        Option(
            "--watch",
            help="Item to watch, in 'type:name' format (e.g. 'model:bert-base-uncased'). Repeatable.",
        ),
    ],
    url: Annotated[
        str | None,
        Option(help="URL to send webhook payloads to. Mutually exclusive with --job-id."),
    ] = None,
    job_id: Annotated[
        str | None,
        Option(
            "--job-id",
            help="ID of a Job to trigger (from job.id) instead of pinging a URL. Mutually exclusive with --url.",
        ),
    ] = None,
    domain: Annotated[
        list[WebhookDomain] | None,
        Option(
            "--domain",
            help="Domain to watch: 'repo' or 'discussions'. Repeatable. Defaults to all domains.",
        ),
    ] = None,
    secret: Annotated[
        str | None,
        Option(help="Optional secret used to sign webhook payloads."),
    ] = None,
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    token: TokenOpt = None,
) -> None:
    """Create a new webhook.

    Provide either --url (to ping a remote server) or --job-id (to trigger a Job), but not both.
    The source Job's secrets are not copied to the webhook: pass them again with --secrets / --secrets-file.
    """
    if url is not None and job_id is not None:
        raise click.BadParameter("Provide either --url or --job-id, not both.")
    if url is None and job_id is None:
        raise click.BadParameter("Provide either --url or --job-id.")
    job_secrets = _parse_job_secrets(secrets, secrets_file, job_id)
    api = get_hf_api(token=token)
    watched_items = _parse_watch(watch)
    domains = [d.value for d in domain] if domain else None
    webhook = api.create_webhook(
        url=url,
        job_id=job_id,
        watched=watched_items,  # type: ignore
        domains=domains,  # type: ignore
        secret=secret,
        secrets=job_secrets,
    )
    out.result("Webhook created", id=webhook.id)


@webhooks_cli.command(
    "update",
    examples=[
        "hf webhooks update abc123 --url https://new-url.com/hook",
        "hf webhooks update abc123 --watch model:gpt2 --domain repo",
        "hf webhooks update abc123 --secret newsecret",
        "hf webhooks update abc123 --job-id 687f911eaea852de79c4a50a --secrets HF_TOKEN",
    ],
)
def webhooks_update(
    webhook_id: Annotated[str, Argument(help="The ID of the webhook to update.")],
    url: Annotated[
        str | None,
        Option(help="New URL to send webhook payloads to."),
    ] = None,
    job_id: Annotated[
        str | None,
        Option(
            "--job-id",
            help="ID of the source Job to trigger (can be the current one). Required with --secrets / --secrets-file.",
        ),
    ] = None,
    watch: Annotated[
        list[str] | None,
        Option(
            "--watch",
            help=(
                "New list of items to watch, in 'type:name' format. "
                "Repeatable. Replaces the entire existing watched list."
            ),
        ),
    ] = None,
    domain: Annotated[
        list[WebhookDomain] | None,
        Option(
            "--domain",
            help="New list of domains to watch: 'repo' or 'discussions'. Repeatable.",
        ),
    ] = None,
    secret: Annotated[
        str | None,
        Option(help="New secret used to sign webhook payloads."),
    ] = None,
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    token: TokenOpt = None,
) -> None:
    """Update an existing webhook. Only provided options are changed.

    Job secrets passed with --secrets replace stored values with the same name; others are kept.
    """
    if url is not None and job_id is not None:
        raise click.BadParameter("Provide either --url or --job-id, not both.")
    job_secrets = _parse_job_secrets(secrets, secrets_file, job_id)
    api = get_hf_api(token=token)
    watched_items = _parse_watch(watch) if watch else None
    domains = [d.value for d in domain] if domain else None
    webhook = api.update_webhook(
        webhook_id,
        url=url,
        job_id=job_id,
        watched=watched_items,  # type: ignore
        domains=domains,  # type: ignore
        secret=secret,
        secrets=job_secrets,
    )
    out.result("Webhook updated", id=webhook.id)


@webhooks_cli.command(
    "enable",
    examples=[
        "hf webhooks enable abc123",
    ],
)
def webhooks_enable(
    webhook_id: Annotated[str, Argument(help="The ID of the webhook to enable.")],
    token: TokenOpt = None,
) -> None:
    """Enable a disabled webhook."""
    api = get_hf_api(token=token)
    webhook = api.enable_webhook(webhook_id)
    out.result("Webhook enabled", id=webhook.id)


@webhooks_cli.command(
    "disable",
    examples=[
        "hf webhooks disable abc123",
    ],
)
def webhooks_disable(
    webhook_id: Annotated[str, Argument(help="The ID of the webhook to disable.")],
    token: TokenOpt = None,
) -> None:
    """Disable an active webhook."""
    api = get_hf_api(token=token)
    webhook = api.disable_webhook(webhook_id)
    out.result("Webhook disabled", id=webhook.id)


@webhooks_cli.command(
    "delete",
    examples=[
        "hf webhooks delete abc123",
        "hf webhooks delete abc123 --yes",
    ],
)
def webhooks_delete(
    webhook_id: Annotated[str, Argument(help="The ID of the webhook to delete.")],
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a webhook permanently."""
    out.confirm(f"Are you sure you want to delete webhook '{webhook_id}'?", yes=yes)
    api = get_hf_api(token=token)
    api.delete_webhook(webhook_id)
    out.result("Webhook deleted", id=webhook_id)
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
"""Contains commands to manage skills for AI assistants."""

import os
import shutil
from pathlib import Path
from typing import Annotated

from click import Command, Context, Group

from huggingface_hub import constants
from huggingface_hub.errors import CLIError

from ..utils import disable_progress_bars
from . import _skills
from ._cli_utils import TokenOpt, _has_local_formatting_option, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import out
from ._skills import DEFAULT_SKILL_ID


_SKILL_DESCRIPTION = (
    "Hugging Face Hub CLI (`hf`) for downloading, uploading, and managing"
    " models, datasets, spaces, buckets, repos, papers, jobs, and more on the Hugging Face Hub."
    " Use when: handling authentication;"
    " managing local cache;"
    " managing Hugging Face Buckets;"
    " running or scheduling jobs on Hugging Face infrastructure;"
    " managing Hugging Face repos;"
    " discussions and pull requests;"
    " browsing models, datasets and spaces;"
    " reading, searching, or browsing academic papers;"
    " managing collections;"
    " querying datasets;"
    " configuring spaces;"
    " setting up webhooks;"
    " or deploying and managing HF Inference Endpoints."
    " Make sure to use this skill whenever the user mentions"
    " 'hf', 'huggingface', 'Hugging Face', 'huggingface-cli', or 'hugging face cli',"
    " or wants to do anything related to the Hugging Face ecosystem and to AI and ML in general."
    " Also use for cloud storage needs like training checkpoints, data pipelines, or agent traces."
    " Use even if the user doesn't explicitly ask for a CLI command."
    " Replaces the deprecated `huggingface-cli`."
)

_SKILL_YAML_PREFIX = f"""\
---
name: hf-cli
description: "{_SKILL_DESCRIPTION}"
---

Install: `curl -LsSf https://hf.co/cli/install.sh | bash -s`.

The Hugging Face Hub CLI tool `hf` is available. IMPORTANT: The `hf` command replaces the deprecated `huggingface-cli` command.

Use `hf --help` to view available functions. Note that auth commands are now all under `hf auth` e.g. `hf auth whoami`.
"""

_SKILL_TIPS = """
## Mounting repos as local filesystems

To mount Hub repositories or buckets as local filesystems — no download, no copy, no waiting — use `hf-mount`. Files are fetched on demand. GitHub: https://github.com/huggingface/hf-mount

Install: `brew install hf-mount`, or download a binary from https://github.com/huggingface/hf-mount/releases

Some command examples:
- `hf-mount start repo openai-community/gpt2 /tmp/gpt2` — mount a repo (read-only)
- `hf-mount start --hf-token $HF_TOKEN bucket myuser/my-bucket /tmp/data` — mount a bucket (read-write)
- `hf-mount status` / `hf-mount stop /tmp/data` — list or unmount

## Tips

- Use `hf <command> --help` for full options, descriptions, usage, and real-world examples
- Authenticate with `HF_TOKEN` env var (recommended) or with `--token`
- Update the CLI with `hf update` (uses the correct command for the detected install method)
"""

# Flags worth explaining in the common-options glossary. Self-explanatory flags
# (--namespace, --yes, --private, …) are omitted even if they appear frequently.
_COMMON_FLAG_ALLOWLIST = {"--token", "--quiet", "--type", "--format", "--revision"}
# Keep token out of inline command signatures to encourage env based auth.
_INLINE_FLAG_EXCLUDE = {"--token"}

_COMMON_FLAG_HELP_OVERRIDES: dict[str, str] = {
    "--format": "Output format: `--format json` (or `--json`) or `--format table` (default).",
    "--token": "Use a User Access Token. Prefer setting `HF_TOKEN` env var instead of passing `--token`.",
}

# Global formatting flags injected into the skill markdown for commands that
# accept them. They aren't real click params on the command (they're consumed
# globally — see ``_consume_format_flags_for_leaf`` in ``_cli_utils.py``) so we
# add them synthetically here.
_GLOBAL_FORMAT_INLINE_FLAGS = ["--format [auto|human|agent|json|quiet]"]
_GLOBAL_COMMON_FLAGS: dict[str, tuple[str, str]] = {
    "--format": ("--format", "Output format."),
    "--quiet": ("-q / --quiet", "Quiet output (one ID per line)."),
}

skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _type_hint(param) -> str:
    """Value hint for an option: enum choices inline as ``[a|b|c]``, otherwise the TYPE name.

    e.g. `--sort [downloads|likes|trending_score]` instead of `--sort CHOICE`.
    """
    choices = getattr(param.type, "choices", None)
    if choices:
        return "[" + "|".join(str(c) for c in choices) + "]"
    return getattr(param.type, "name", "").upper() or "VALUE"


def _format_params(cmd: Command) -> str:
    """Format required params: positional as UPPER_CASE, options as ``--name TYPE``."""
    parts = []
    for p in cmd.params:
        if not p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = next((o for o in getattr(p, "opts", []) if o.startswith("--")), None)
        if long_name is not None:
            type_name = _type_hint(p)
            parts.append(f"{long_name} {type_name}")
        elif p.name:
            parts.append(p.human_readable_name)
    return " ".join(parts)


def _collect_leaf_commands(group: Group, ctx: Context, path_parts: list[str]) -> list[tuple[list[str], Command]]:
    """Recursively walk a Click Group, returning (full_path_parts, cmd) for every leaf command."""
    leaves: list[tuple[list[str], Command]] = []
    sub_ctx = Context(group, parent=ctx, info_name=path_parts[-1])
    for name in group.list_commands(sub_ctx):
        cmd = group.get_command(sub_ctx, name)
        if cmd is None or cmd.hidden:
            continue
        if materialize := getattr(cmd, "materialize", None):
            cmd = materialize()
        assert cmd is not None
        child_path = [*path_parts, name]
        if isinstance(cmd, Group):
            leaves.extend(_collect_leaf_commands(cmd, sub_ctx, child_path))
        else:
            leaves.append((child_path, cmd))
    return leaves


def _iter_optional_params(cmd: Command):
    """Yield (param, long_name, short_name) for each optional, non-internal param."""
    for p in cmd.params:
        if p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = None
        short_name = None
        for opt in getattr(p, "opts", []):
            if opt.startswith("--"):
                long_name = long_name or opt
            elif opt.startswith("-"):
                short_name = opt
        if long_name:
            yield p, long_name, short_name


def _accepts_global_format_flags(cmd: Command) -> bool:
    """Return True if the leaf command accepts the global '--format' / '--json' / '-q' flags."""
    if cmd.context_settings.get("ignore_unknown_options"):
        return False
    return not _has_local_formatting_option(cmd)


def _get_flag_names(cmd: Command, *, exclude: set[str] | None = None) -> list[str]:
    """Return long-form flag names (--foo) for optional, non-internal params.

    Boolean flags are bare ('--dry-run').  Value-taking options include a type hint ('--include TEXT', '--max-workers INTEGER').
    Synthetic global formatting flags are appended for commands that accept them.
    """
    flags: list[str] = []
    for p, long_name, _short in _iter_optional_params(cmd):
        if exclude and long_name in exclude:
            continue
        if getattr(p, "is_flag", False):
            flags.append(long_name)
        else:
            type_name = _type_hint(p)
            flags.append(f"{long_name} {type_name}")
    if _accepts_global_format_flags(cmd):
        flags.extend(flag for flag in _GLOBAL_FORMAT_INLINE_FLAGS if not (exclude and flag.split()[0] in exclude))
    return flags


def _compute_common_flags(
    leaf_commands: list[tuple[list[str], Command]],
) -> dict[str, tuple[str, str]]:
    """Collect display info for flags in the allowlist."""
    flag_info: dict[str, tuple[str, str]] = {}

    for _path, cmd in leaf_commands:
        for p, long_name, short_name in _iter_optional_params(cmd):
            if long_name not in _COMMON_FLAG_ALLOWLIST:
                continue
            # Prefer the version with a short form (e.g. "-q / --quiet" over just "--quiet")
            if long_name not in flag_info or (short_name and " / " not in flag_info[long_name][0]):
                display = f"{short_name} / {long_name}" if short_name else long_name
                help_text = (getattr(p, "help", None) or "").split("\n")[0].strip()
                flag_info[long_name] = (display, help_text)

    # Inject the global formatting flags as common flags whenever any leaf
    # command accepts them (the vast majority do).
    if any(_accepts_global_format_flags(cmd) for _path, cmd in leaf_commands):
        for long_name, entry in _GLOBAL_COMMON_FLAGS.items():
            flag_info.setdefault(long_name, entry)

    return flag_info


def _render_leaf(path_parts: list[str], cmd: Command) -> str:
    """Render a single leaf command as a markdown list entry."""
    help_text = (cmd.help or "").split("\n")[0].strip()
    params = _format_params(cmd)
    parts = ["hf", *path_parts] + ([params] if params else [])
    entry = f"- `{' '.join(parts)}` — {help_text}"
    flags = _get_flag_names(cmd, exclude=_INLINE_FLAG_EXCLUDE)
    if flags:
        entry += f" `[{' '.join(flags)}]`"
    return entry


def build_skill_md() -> str:
    # Lazy import to avoid circular dependency (hf.py imports skills_cli from this module)
    from huggingface_hub import __version__
    from huggingface_hub.cli.hf import app

    click_app = app  # the app is already a click.Group
    ctx = Context(click_app, info_name="hf")

    top_level: list[tuple[list[str], Command]] = []
    groups: list[tuple[str, Group]] = []
    for name in sorted(click_app.list_commands(ctx)):  # type: ignore[attr-defined]
        cmd = click_app.get_command(ctx, name)  # type: ignore[attr-defined]
        if cmd is None or cmd.hidden:
            continue
        if materialize := getattr(cmd, "materialize", None):
            cmd = materialize()
        assert cmd is not None
        if isinstance(cmd, Group):
            groups.append((name, cmd))
        else:
            top_level.append(([name], cmd))

    group_leaves: list[tuple[str, list[tuple[list[str], Command]]]] = []
    all_leaf_commands: list[tuple[list[str], Command]] = list(top_level)
    for name, group in groups:
        leaves = _collect_leaf_commands(group, ctx, [name])
        group_leaves.append((name, leaves))
        all_leaf_commands.extend(leaves)

    common_flags = _compute_common_flags(all_leaf_commands)

    # wrap in list to widen list[LiteralString] -> list[str] for `ty`
    lines: list[str] = list(_SKILL_YAML_PREFIX.splitlines())
    lines.append("")
    lines.append(f"Generated with `huggingface_hub v{__version__}`. Run `hf skills add --force` to regenerate.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")

    for path_parts, cmd in top_level:
        lines.append(_render_leaf(path_parts, cmd))

    groups_dict = dict(groups)
    for name, leaves in group_leaves:
        group_cmd = groups_dict[name]
        help_text = (group_cmd.help or "").split("\n")[0].strip()
        lines.append("")
        lines.append(f"### `hf {name}` — {help_text}")
        lines.append("")
        for path_parts, cmd in leaves:
            lines.append(_render_leaf(path_parts, cmd))

    if common_flags:
        lines.append("")
        lines.append("## Common options")
        lines.append("")
        for long_name, (display, help_text) in sorted(common_flags.items()):
            help_text = _COMMON_FLAG_HELP_OVERRIDES.get(long_name, help_text)
            if help_text:
                lines.append(f"- `{display}` — {help_text}")
            else:
                lines.append(f"- `{display}`")

    lines.extend(_SKILL_TIPS.splitlines())

    return "\n".join(lines)


def _remove_existing(path: Path, force: bool) -> None:
    """Remove existing file/directory/symlink if force is True, otherwise raise an error."""
    if not (path.exists() or path.is_symlink()):
        return
    if not force:
        raise CLIError(f"Skill already exists at {path}.\nRe-run with --force to overwrite.")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_to(skills_dir: Path, skill_name: str, force: bool) -> Path:
    """Install a marketplace skill into a skills directory. Returns the installed path."""
    try:
        if skill_name.strip() == DEFAULT_SKILL_ID:
            return _skills.install_generated_skill(build_skill_md(), skills_dir, force=force)
        return _skills.add_skill(skill_name, skills_dir, force=force)
    except FileExistsError as exc:
        raise CLIError(f"{exc}\nRe-run with --force to overwrite.") from exc


def _create_symlink(agent_skills_dir: Path, skill_name: str, central_skill_path: Path, force: bool) -> Path:
    """Create a relative symlink from agent directory to the central skill location."""
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / skill_name

    _remove_existing(link_path, force)
    try:
        link_path.symlink_to(os.path.relpath(central_skill_path, agent_skills_dir))
    except OSError:
        # Windows needs Developer Mode or admin rights for symlinks.
        # Fall back to a copy: `hf skills update` walks both roots, so the copy stays in sync.
        shutil.copytree(central_skill_path, link_path)

    return link_path


def _resolve_update_roots(*, global_: bool, dest: Path | None) -> list[Path]:
    if dest is not None:
        if global_:
            raise CLIError("--dest cannot be combined with --global.")
        return [dest.expanduser().resolve()]

    if global_:
        roots = [constants.AGENTS_SKILLS_GLOBAL_PATH, constants.CLAUDE_SKILLS_GLOBAL_PATH]
    else:
        roots = [constants.AGENTS_SKILLS_LOCAL_PATH, constants.CLAUDE_SKILLS_LOCAL_PATH]
    return [root.expanduser().resolve() for root in roots]


@skills_cli.command("preview")
def skills_preview() -> None:
    """Print the generated `hf-cli` SKILL.md to stdout."""
    print(build_skill_md())


@skills_cli.command(
    "list | ls",
    examples=[
        "hf skills list",
        "hf skills list --format json",
    ],
)
def skills_list(
    token: TokenOpt = None,
) -> None:
    """List available skills from the Hugging Face marketplace."""
    install_locations: list[tuple[str, Path]] = [
        ("project", constants.AGENTS_SKILLS_LOCAL_PATH),
        ("project (claude)", constants.CLAUDE_SKILLS_LOCAL_PATH),
        ("global", constants.AGENTS_SKILLS_GLOBAL_PATH),
        ("global (claude)", constants.CLAUDE_SKILLS_GLOBAL_PATH),
    ]
    installed: dict[str, set[str]] = {}
    for label, root in install_locations:
        for skill_dir in _skills._iter_unique_skill_dirs([root]):
            installed.setdefault(skill_dir.name.lower(), set()).add(label)

    api = get_hf_api(token=token)
    with disable_progress_bars():
        skills = _skills._load_marketplace_skills(api)
    results = [
        {
            "name": skill.name,
            "description": skill.description or "",
            **{
                label: "yes" if label in installed.get(skill.name.lower(), set()) else ""
                for label, _ in install_locations
            },
        }
        for skill in skills
    ]
    out.table(
        results,
        id_key="name",
        alignments={"project": "right", "global": "right", "project (claude)": "right", "global (claude)": "right"},
    )


@skills_cli.command(
    "add",
    examples=[
        "hf skills add",
        "hf skills add huggingface-gradio --dest=~/my-skills",
        "hf skills add --global",
        "hf skills add huggingface-gradio --global",
    ],
)
def skills_add(
    name: Annotated[
        str,
        Argument(help="Marketplace skill name.", show_default=False),
    ] = DEFAULT_SKILL_ID,
    global_: Annotated[
        bool,
        Option(
            "--global",
            "-g",
            help="Install globally (user-level) instead of in the current project directory.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        Option(
            help="Install into a custom destination (path to skills directory).",
        ),
    ] = None,
    force: Annotated[
        bool,
        Option(
            "--force",
            help="Overwrite existing skills in the destination.",
        ),
    ] = False,
) -> None:
    """Install a Hugging Face skill for an AI assistant.

    The default `hf-cli` skill is generated locally from the installed CLI version;
    other skills are downloaded from the Hugging Face marketplace.
    Default location is in the current directory (.agents/skills) or user-level (~/.agents/skills).
    The skill is also symlinked into Claude Code's skills directory (`.claude/skills` or `~/.claude/skills`,
    honoring `CLAUDE_CONFIG_DIR` when set), unless `--dest` is used.
    """
    if dest is not None:
        if global_:
            raise CLIError("--dest cannot be combined with --global.")
        skill_dest = _install_to(dest, name, force)
        print(f"Installed '{name}' to {skill_dest}")
        return

    # Install to central location, then expose it to Claude Code which only reads `.claude/skills`
    central_path = constants.AGENTS_SKILLS_GLOBAL_PATH if global_ else constants.AGENTS_SKILLS_LOCAL_PATH
    central_skill_path = _install_to(central_path, name, force)
    print(f"Installed '{name}' to central location: {central_skill_path}")

    claude_path = constants.CLAUDE_SKILLS_GLOBAL_PATH if global_ else constants.CLAUDE_SKILLS_LOCAL_PATH
    link_path = _create_symlink(claude_path, name, central_skill_path, force)
    print(f"Created symlink: {link_path}" if link_path.is_symlink() else f"Copied '{name}' to {link_path}")
    if not global_:
        out.hint(f"Run `hf skills add {name} --global` to install the skill globally instead.")


@skills_cli.command(
    "update",
    examples=[
        "hf skills update",
        "hf skills update hf-cli",
        "hf skills update huggingface-gradio --dest=~/my-skills",
        "hf skills update -g",
    ],
)
def skills_update(
    name: Annotated[
        str | None,
        Argument(help="Optional installed skill name to update.", show_default=False),
    ] = None,
    global_: Annotated[
        bool,
        Option(
            "--global",
            "-g",
            help="Use global skills directories instead of the current project.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        Option(
            help="Update skills in a custom skills directory.",
        ),
    ] = None,
) -> None:
    """Update installed Hugging Face marketplace skills."""
    roots = _resolve_update_roots(global_=global_, dest=dest)

    results = _skills.update_skills(roots, selector=name, hf_cli_content=build_skill_md())
    if not results:
        print("No installed skills found.")
    for result in results:
        detail = f" ({result.detail})" if result.detail else ""
        print(f"{result.name}: {result.status}{detail}")

    if not global_ and dest is None:
        command = f"hf skills update {name} --global" if name else "hf skills update --global"
        out.hint(f"Run `{command}` to update globally installed skills instead.")
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
"""Contains commands to interact with repositories on the Hugging Face Hub."""

import enum
from typing import Annotated, cast

from huggingface_hub import SpaceHardware
from huggingface_hub.cli._cli_utils import SoftChoice
from huggingface_hub.errors import CLIError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import REPO_REGIONS

from ._city_game import run_city_game
from ._cli_utils import (
    REPO_LIST_DEFAULT_LIMIT,
    EnvFileOpt,
    EnvOpt,
    LimitOpt,
    PrivateOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOpt,
    RevisionOpt,
    SearchOpt,
    SecretsFileOpt,
    SecretsOpt,
    TokenOpt,
    VolumesOpt,
    env_map_to_key_value_list,
    get_hf_api,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._cp import make_cp
from ._file_listing import format_size
from ._framework import Argument, Option
from ._output import OutputFormat, out


repos_cli = typer_factory(help="Manage repos on the Hub.")


class RepoTypeAll(str, enum.Enum):
    model = "model"
    dataset = "dataset"
    space = "space"
    bucket = "bucket"


class GatedChoices(str, enum.Enum):
    auto = "auto"
    manual = "manual"
    false = "false"


PublicOpt = Annotated[
    bool | None,
    Option(
        "--public",
        help="Whether to make the repo public. Ignored if the repo already exists.",
    ),
]

ProtectedOpt = Annotated[
    bool | None,
    Option(
        "--protected",
        help="Whether to make the Space protected (Spaces only). Ignored if the repo already exists.",
    ),
]
SpaceHardwareOpt = Annotated[
    str | None,
    Option(
        "--flavor",
        help="Space hardware flavor (e.g. 'cpu-basic', 't4-medium', 'l4x4'). Only for Spaces.",
        click_type=SoftChoice(SpaceHardware),
    ),
]

SpaceSleepTimeOpt = Annotated[
    int | None,
    Option(
        "--sleep-time",
        help="Seconds of inactivity before the Space is put to sleep. Use -1 to disable. Only for Spaces.",
    ),
]


tag_cli = typer_factory(help="Manage tags for a repo on the Hub.")
branch_cli = typer_factory(help="Manage branches for a repo on the Hub.")
repos_cli.add_group(tag_cli, name="tag")
repos_cli.add_group(branch_cli, name="branch")


@repos_cli.command(
    "list | ls",
    examples=[
        "hf repos ls",
        "hf repos ls --explore",
        "hf repos ls --namespace my-org --search bert",
    ],
)
def repo_list(
    namespace: Annotated[
        str | None,
        Option(
            help="Organization name. If not provided, lists repos for the authenticated user.",
        ),
    ] = None,
    repo_type: Annotated[
        RepoTypeAll | None,
        Option(
            "--type",
            "--repo-type",
            help="Filter by repository type (model, dataset, space, or bucket).",
        ),
    ] = None,
    search: SearchOpt = None,
    limit: LimitOpt = REPO_LIST_DEFAULT_LIMIT,
    explore: Annotated[
        bool,
        Option("--explore", help="Explore your repos as an interactive 3D city."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """List all repos (models, datasets, spaces, buckets) with storage info."""
    api = get_hf_api(token=token)
    repos = list(api.list_user_repos(namespace=namespace))
    if repo_type is not None:
        repos = [r for r in repos if r.type == repo_type.value]
    if search is not None:
        search_lower = search.lower()
        repos = [r for r in repos if search_lower in r.id.lower()]
    total = len(repos)

    if explore:
        if out.mode == OutputFormat.human:
            run_city_game(repos)
            return
        raise CLIError("Repository exploration is only available in terminal.")

    if limit > 0:
        repos = repos[:limit]
    items = [
        {
            "id": r.id,
            "type": r.type,
            "updated": r.updated_at.strftime("%Y-%m-%d"),
            "visibility": r.visibility,
            "storage": format_size(r.storage, human_readable=True),
            "%_of_total": f"{r.storage_percent:.1f}%",
        }
        for r in repos
    ]
    out.table(items, id_key="id", alignments={"storage": "right", "%_of_total": "right"})
    if limit > 0 and total > limit:
        out.hint(f"Showing {limit} of {total} repos. Use `--limit 0` to list all.")


@repos_cli.command(
    "create",
    examples=[
        "hf repos create my-model",
        "hf repos create my-dataset --repo-type dataset --private",
        "hf repos create my-space --type space --sdk gradio --flavor t4-medium --secrets HF_TOKEN -e THEME=dark --protected",
        "hf repos create my-jupyterlab --type space --template SpacesExamples/jupyterlab",
        "hf repos create my-space --type space --sdk gradio -v hf://org/my-model:/models -v hf://buckets/org/b:/data",
        "hf repos create my-model --region us",
    ],
)
def repo_create(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    sdk: Annotated[
        str | None,
        Option(
            "--sdk",
            "--space-sdk",
            help="Hugging Face Spaces SDK type. Required when --type is set to 'space'.",
        ),
    ] = None,
    template: Annotated[
        str | None,
        Option(
            "--template",
            help=(
                "Create a Space from an official template. Pass a template repo id (e.g. "
                "'SpacesExamples/jupyterlab') or its short name (e.g. 'JupyterLab'). List available templates with "
                "`hf spaces templates`. Spaces only."
            ),
        ),
    ] = None,
    private: PrivateOpt = None,
    public: PublicOpt = None,
    protected: ProtectedOpt = None,
    token: TokenOpt = None,
    exist_ok: Annotated[
        bool,
        Option(
            help="Do not raise an error if repo already exists.",
        ),
    ] = False,
    resource_group_id: Annotated[
        str | None,
        Option(
            help="Resource group in which to create the repo. Resource groups is only available for Enterprise Hub organizations.",
        ),
    ] = None,
    region: Annotated[
        REPO_REGIONS | None,
        Option(
            "--region",
            help="Cloud region in which to create the repo. Can be one of 'us' or 'eu'. Requires Team plan or above.",
        ),
    ] = None,
    hardware: SpaceHardwareOpt = None,
    sleep_time: SpaceSleepTimeOpt = None,
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    volume: VolumesOpt = None,
) -> None:
    """Create a new repo on the Hub."""
    api = get_hf_api(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type.value,
        visibility="private" if private else "public" if public else "protected" if protected else None,  # type: ignore [arg-type]
        token=token,
        exist_ok=exist_ok,
        resource_group_id=resource_group_id,
        region=region,
        space_sdk=sdk,
        space_hardware=cast(SpaceHardware | None, hardware),
        space_sleep_time=sleep_time,
        space_secrets=env_map_to_key_value_list(parse_env_map(secrets, secrets_file)),
        space_variables=env_map_to_key_value_list(parse_env_map(env, env_file)),
        space_volumes=parse_volumes(volume),
        space_template=template,
    )
    out.result("Repo created", repo_id=repo_url.repo_id, url=str(repo_url))


@repos_cli.command(
    "duplicate",
    examples=[
        "hf repos duplicate openai/gdpval --type dataset",
        "hf repos duplicate multimodalart/dreambooth-training my-dreambooth --type space --flavor l4x4 --secrets HF_TOKEN --private",
        "hf repos duplicate org/my-space my-space --type space -v hf://org/my-model:/models -v hf://buckets/org/b:/data",
    ],
)
def repo_duplicate(
    from_id: RepoIdArg,
    to_id: Annotated[
        str | None,
        Argument(
            help="Destination repo ID (e.g. `myorg/my-copy`). Defaults to your namespace with the same repo name.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    private: PrivateOpt = None,
    public: PublicOpt = None,
    protected: ProtectedOpt = None,
    token: TokenOpt = None,
    exist_ok: Annotated[
        bool,
        Option(
            help="Do not raise an error if repo already exists.",
        ),
    ] = False,
    resource_group_id: Annotated[
        str | None,
        Option(
            help="Resource group in which to create the new repo. Resource groups is only available for Enterprise Hub organizations.",
        ),
    ] = None,
    hardware: SpaceHardwareOpt = None,
    sleep_time: SpaceSleepTimeOpt = None,
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    volume: VolumesOpt = None,
) -> None:
    """Duplicate a repo on the Hub (model, dataset, or Space)."""
    api = get_hf_api(token=token)
    repo_url = api.duplicate_repo(
        from_id=from_id,
        to_id=to_id,
        repo_type=repo_type.value,
        visibility="private" if private else "public" if public else "protected" if protected else None,  # type: ignore [arg-type]
        token=token,
        exist_ok=exist_ok,
        resource_group_id=resource_group_id,
        space_hardware=cast(SpaceHardware | None, hardware),
        space_sleep_time=sleep_time,
        space_secrets=env_map_to_key_value_list(parse_env_map(secrets, secrets_file)),
        space_variables=env_map_to_key_value_list(parse_env_map(env, env_file)),
        space_volumes=parse_volumes(volume),
    )
    out.result("Repo duplicated", from_id=from_id, to_id=repo_url.repo_id, url=str(repo_url))
    if repo_url.files_copy_pending:
        out.warning(
            f"LFS/Xet files are still being copied in the background. Downloading them from '{repo_url.repo_id}'"
            " may fail until the copy is complete."
        )


@repos_cli.command("delete", examples=["hf repos delete my-model"])
def repo_delete(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
    missing_ok: Annotated[
        bool,
        Option(
            help="If set to True, do not raise an error if repo does not exist.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically.",
        ),
    ] = False,
) -> None:
    """Delete a repo from the Hub. This is an irreversible operation."""
    out.confirm(f"You are about to permanently delete {repo_type.value} '{repo_id}'. Proceed?", yes=yes)
    api = get_hf_api(token=token)
    api.delete_repo(
        repo_id=repo_id,
        repo_type=repo_type.value,
        missing_ok=missing_ok,
    )
    out.result("Repo deleted", repo_id=repo_id)


@repos_cli.command("move", examples=["hf repos move old-namespace/my-model new-namespace/my-model"])
def repo_move(
    from_id: RepoIdArg,
    to_id: RepoIdArg,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Move a repository from a namespace to another namespace."""
    api = get_hf_api(token=token)
    api.move_repo(
        from_id=from_id,
        to_id=to_id,
        repo_type=repo_type.value,
    )
    out.result("Repo moved", from_id=from_id, to_id=to_id)


@repos_cli.command(
    "settings",
    examples=[
        "hf repos settings my-model --private",
        "hf repos settings my-model --gated auto",
        "hf repos settings my-space --repo-type space --protected",
    ],
)
def repo_settings(
    repo_id: RepoIdArg,
    gated: Annotated[
        GatedChoices | None,
        Option(
            help="The gated status for the repository.",
        ),
    ] = None,
    private: PrivateOpt = None,
    public: PublicOpt = None,
    protected: ProtectedOpt = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Update the settings of a repository."""
    api = get_hf_api(token=token)
    api.update_repo_settings(
        repo_id=repo_id,
        gated=(None if gated is None else False if gated is GatedChoices.false else gated.value),
        visibility="private" if private else "public" if public else "protected" if protected else None,  # type: ignore [arg-type]
        repo_type=repo_type.value,
    )
    out.result("Repo settings updated", repo_id=repo_id)


@repos_cli.command(
    "delete-files",
    examples=[
        "hf repos delete-files my-model file.txt",
        'hf repos delete-files my-model "*.json"',
        "hf repos delete-files my-model folder/",
    ],
)
def repo_delete_files(
    repo_id: RepoIdArg,
    patterns: Annotated[
        list[str],
        Argument(
            help="Glob patterns to match files to delete. Based on fnmatch, '*' matches files recursively.",
        ),
    ],
    repo_type: RepoTypeOpt = RepoType.model,
    revision: RevisionOpt = None,
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
            help="Whether to create a new Pull Request for these changes.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete files from a repo on the Hub."""
    api = get_hf_api(token=token)
    url = api.delete_files(
        delete_patterns=patterns,
        repo_id=repo_id,
        repo_type=repo_type.value,
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )
    out.result("Files deleted", repo_id=repo_id, commit_url=url)


# `hf repos cp` is an alias for the top-level `hf cp` command (see `cli/_cp.py`).
repos_cli.command(
    name="cp",
    examples=[
        # Download (repo or bucket -> local / stdout)
        "hf repos cp hf://username/my-model/config.json config.json",
        "hf repos cp hf://datasets/username/my-dataset/data.csv data/",
        "hf repos cp hf://username/my-model/config.json -",
        # Upload (local / stdin -> repo)
        "hf repos cp model.safetensors hf://username/my-model/model.safetensors",
        "hf repos cp config.json hf://username/my-model/logs/",
        "hf repos cp - hf://username/my-model/config.json",
        # Remote to remote (repo -> repo)
        "hf repos cp hf://username/source-model/config.json hf://username/dest-model/config.json",
        "hf repos cp hf://datasets/username/my-dataset/processed/ hf://datasets/username/dest-dataset/processed/",
        "hf repos cp hf://username/my-model/logs/ hf://username/archive-model/logs/",
    ],
)(make_cp("repos"))


@branch_cli.command(
    "create",
    examples=[
        "hf repos branch create my-model dev",
        "hf repos branch create my-model dev --revision abc123",
    ],
)
def branch_create(
    repo_id: RepoIdArg,
    branch: Annotated[
        str,
        Argument(
            help="The name of the branch to create.",
        ),
    ],
    revision: RevisionOpt = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
    exist_ok: Annotated[
        bool,
        Option(
            help="If set to True, do not raise an error if branch already exists.",
        ),
    ] = False,
) -> None:
    """Create a new branch for a repo on the Hub."""
    api = get_hf_api(token=token)
    api.create_branch(
        repo_id=repo_id,
        branch=branch,
        revision=revision,
        repo_type=repo_type.value,
        exist_ok=exist_ok,
    )
    out.result("Branch created", branch=branch, repo_type=repo_type.value, repo_id=repo_id)


@branch_cli.command("delete", examples=["hf repos branch delete my-model dev"])
def branch_delete(
    repo_id: RepoIdArg,
    branch: Annotated[
        str,
        Argument(
            help="The name of the branch to delete.",
        ),
    ],
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Delete a branch from a repo on the Hub."""
    api = get_hf_api(token=token)
    api.delete_branch(
        repo_id=repo_id,
        branch=branch,
        repo_type=repo_type.value,
    )
    out.result("Branch deleted", branch=branch, repo_type=repo_type.value, repo_id=repo_id)


@tag_cli.command(
    "create",
    examples=[
        "hf repos tag create my-model v1.0",
        'hf repos tag create my-model v1.0 -m "First release"',
    ],
)
def tag_create(
    repo_id: RepoIdArg,
    tag: Annotated[
        str,
        Argument(
            help="The name of the tag to create.",
        ),
    ],
    message: Annotated[
        str | None,
        Option(
            "-m",
            "--message",
            help="The description of the tag to create.",
        ),
    ] = None,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Create a tag for a repo."""
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    try:
        api.create_tag(repo_id=repo_id, tag=tag, tag_message=message, revision=revision, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found.") from e
    except HfHubHTTPError as e:
        if e.response.status_code == 409:
            raise CLIError(f"Tag '{tag}' already exists on '{repo_id}'.") from e
        raise
    out.result("Tag created", tag=tag, repo_type=repo_type_str, repo_id=repo_id)


@tag_cli.command("list | ls", examples=["hf repos tag list my-model"])
def tag_list(
    repo_id: RepoIdArg,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """List tags for a repo."""
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    items = [{"name": t.name, "target_commit": t.target_commit, "ref": t.ref} for t in refs.tags]
    out.table(items)


@tag_cli.command("delete", examples=["hf repos tag delete my-model v1.0"])
def tag_delete(
    repo_id: RepoIdArg,
    tag: Annotated[
        str,
        Argument(
            help="The name of the tag to delete.",
        ),
    ],
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically",
        ),
    ] = False,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Delete a tag for a repo."""
    repo_type_str = repo_type.value
    out.text(f"You are about to delete tag {tag} on {repo_type_str} {repo_id}")
    out.confirm("Proceed?", yes=yes)
    api = get_hf_api(token=token)
    try:
        api.delete_tag(repo_id=repo_id, tag=tag, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Tag '{tag}' not found on '{repo_id}'.") from e
    out.result("Tag deleted", tag=tag, repo_type=repo_type_str, repo_id=repo_id)
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
"""Contains commands to interact with datasets on the Hugging Face Hub."""

import enum
from typing import Annotated, get_args

import click

from huggingface_hub._dataset_viewer import execute_raw_sql_query
from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import DatasetSort_T, ExpandDatasetProperty_T
from huggingface_hub.repocard import DatasetCard

from ._cli_utils import (
    REPO_LIST_DEFAULT_LIMIT,
    AuthorOpt,
    FilterOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    get_hf_api,
    make_expand_properties_parser,
    typer_factory,
)
from ._file_listing import list_repo_files_cmd
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


_EXPAND_PROPERTIES = sorted(get_args(ExpandDatasetProperty_T))
_SORT_OPTIONS = get_args(DatasetSort_T)
DatasetSortEnum = enum.Enum("DatasetSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


datasets_cli = typer_factory(help="Interact with datasets on the Hub.")


@datasets_cli.command(
    "list | ls",
    examples=[
        "hf datasets ls",
        "hf datasets ls --sort downloads --limit 10",
        'hf datasets ls --search "code"',
        "hf datasets ls --filter benchmark:official",
        "hf datasets ls HuggingFaceFW/fineweb",
        "hf datasets ls HuggingFaceFW/fineweb -R",
        "hf datasets ls HuggingFaceFW/fineweb --tree -h",
    ],
)
def datasets_ls(
    repo_id: Annotated[
        str | None,
        Argument(help="Dataset ID (e.g. `username/repo-name`) to list files from. If omitted, lists datasets."),
    ] = None,
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        DatasetSortEnum | None,
        Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = REPO_LIST_DEFAULT_LIMIT,
    expand: ExpandOpt = None,
    human_readable: Annotated[
        bool,
        Option("--human-readable", "-h", help="Show sizes in human readable format (only for listing files)."),
    ] = False,
    as_tree: Annotated[
        bool,
        Option("--tree", help="List files in tree format (only for listing files)."),
    ] = False,
    recursive: Annotated[
        bool,
        Option("--recursive", "-R", help="List files recursively (only for listing files)."),
    ] = False,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
) -> None:
    """List datasets on the Hub, or files in a dataset repo.

    When called with no argument, lists datasets on the Hub.
    When called with a dataset ID, lists files in that dataset repo.
    """
    if repo_id is not None:
        if search is not None:
            raise click.BadParameter("Cannot use --search when listing files.")
        if author is not None:
            raise click.BadParameter("Cannot use --author when listing files.")
        if filter is not None:
            raise click.BadParameter("Cannot use --filter when listing files.")
        if sort is not None:
            raise click.BadParameter("Cannot use --sort when listing files.")
        if limit != REPO_LIST_DEFAULT_LIMIT:
            raise click.BadParameter("Cannot use --limit when listing files.")
        if expand is not None:
            raise click.BadParameter("Cannot use --expand when listing files.")
        return list_repo_files_cmd(
            repo_id=repo_id,
            repo_type="dataset",
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            revision=revision,
            token=token,
        )

    if as_tree:
        raise click.BadParameter("Cannot use --tree when listing datasets.")
    if recursive:
        raise click.BadParameter("Cannot use --recursive when listing datasets.")
    if human_readable:
        raise click.BadParameter("Cannot use --human-readable when listing datasets.")
    if revision is not None:
        raise click.BadParameter("Cannot use --revision when listing datasets.")

    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        _dataclass_to_dict(dataset_info)
        for dataset_info in api.list_datasets(
            filter=filter,
            author=author,
            search=search,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore
        )
    ]
    out.table(results)


@datasets_cli.command(
    "leaderboard",
    examples=[
        "hf datasets leaderboard SWE-bench/SWE-bench_Verified",
        "hf datasets leaderboard SWE-bench/SWE-bench_Verified --limit 5 --format json",
        "hf datasets ls --filter benchmark:official  # list available leaderboards",
    ],
)
def datasets_leaderboard(
    dataset_id: Annotated[str, Argument(help="The benchmark dataset ID (e.g. `SWE-bench/SWE-bench_Verified`).")],
    limit: LimitOpt = 20,
    token: TokenOpt = None,
) -> None:
    """List model scores from a dataset leaderboard. This command helps find the best models for a task or compare models by benchmark scores. Use 'hf datasets ls --filter benchmark:official' to list available leaderboards."""
    api = get_hf_api(token=token)
    leaderboard = api.get_dataset_leaderboard(repo_id=dataset_id)
    results = [_dataclass_to_dict(entry) for entry in leaderboard[:limit]]
    out.table(
        results,
        headers=["rank", "model_id", "value", "source"],
        id_key="model_id",
    )
    out.hint("Use 'hf datasets ls --filter benchmark:official' to list available leaderboards.")
    if leaderboard:
        out.hint(f"Use 'hf models info {leaderboard[0].model_id}' to get details about a model.")


@datasets_cli.command(
    "info",
    examples=[
        "hf datasets info HuggingFaceFW/fineweb",
        "hf datasets info my-dataset --expand downloads,likes,tags",
    ],
)
def datasets_info(
    dataset_id: Annotated[str, Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a dataset on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.dataset_info(repo_id=dataset_id, revision=revision, expand=expand)  # type: ignore
    except RepositoryNotFoundError as e:
        raise CLIError(f"Dataset '{dataset_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{dataset_id}'.") from e
    out.dict(info)


@datasets_cli.command(
    "parquet",
    examples=[
        "hf datasets parquet cfahlgren1/hub-stats",
        "hf datasets parquet cfahlgren1/hub-stats --subset models",
        "hf datasets parquet cfahlgren1/hub-stats --split train",
        "hf datasets parquet cfahlgren1/hub-stats --format json",
    ],
)
def datasets_parquet(
    dataset_id: Annotated[str, Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    subset: Annotated[str | None, Option("--subset", help="Filter parquet entries by subset/config.")] = None,
    split: Annotated[str | None, Option(help="Filter parquet entries by split.")] = None,
    token: TokenOpt = None,
) -> None:
    """List parquet file URLs available for a dataset."""
    api = get_hf_api(token=token)
    entries = api.list_dataset_parquet_files(repo_id=dataset_id, config=subset)
    filtered = [entry for entry in entries if split is None or entry.split == split]
    results = [
        {"subset": entry.config, "split": entry.split, "url": entry.url, "size": entry.size} for entry in filtered
    ]
    out.table(results, headers=["subset", "split", "url", "size"], id_key="url")


@datasets_cli.command(
    "sql",
    examples=[
        "hf datasets sql \"SELECT COUNT(*) AS rows FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet')\"",
        "hf datasets sql \"SELECT * FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet') LIMIT 5\" --format json",
    ],
)
def datasets_sql(
    sql: Annotated[str, Argument(help="Raw SQL query to execute.")],
    token: TokenOpt = None,
) -> None:
    """Execute a raw SQL query with DuckDB against dataset parquet URLs."""
    try:
        result = execute_raw_sql_query(sql_query=sql, token=token)
    except ImportError as e:
        raise CLIError(str(e)) from e
    out.table(result)


@datasets_cli.command(
    "card",
    examples=[
        "hf datasets card HuggingFaceFW/fineweb",
        "hf datasets card HuggingFaceFW/fineweb --metadata",
        "hf datasets card HuggingFaceFW/fineweb --metadata --format json",
        "hf datasets card HuggingFaceFW/fineweb --text",
    ],
)
def datasets_card(
    dataset_id: Annotated[str, Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    metadata: Annotated[bool, Option("--metadata", help="Output only the metadata from the card.")] = False,
    text: Annotated[bool, Option("--text", help="Output only the text body (no metadata).")] = False,
    token: TokenOpt = None,
) -> None:
    """Get the dataset card (README) for a dataset on the Hub."""
    if metadata and text:
        raise CLIError("--metadata and --text are mutually exclusive.")
    card = DatasetCard.load(dataset_id, token=token)
    if metadata:
        out.dict(card.data.to_dict())
    elif text:
        out.text(card.text)
    else:
        out.text(card.content)
        out.hint(f"Use `hf datasets card {dataset_id} --metadata` to extract only the card metadata.")
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
"""Contains commands to interact with spaces on the Hugging Face Hub."""

import enum
import functools
import itertools
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

import click
from packaging import version
from typing_extensions import assert_never

from huggingface_hub._hot_reload.client import multi_replica_reload_events
from huggingface_hub._hot_reload.types import ApiGetReloadEventSourceData, ReloadRegion
from huggingface_hub._space_api import SpaceHardware, SpaceStage
from huggingface_hub.cli._cli_utils import SoftChoice
from huggingface_hub.errors import CLIError, RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import ExpandSpaceProperty_T, HfApi, SpaceSort_T
from huggingface_hub.repocard import SpaceCard
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils._parsing import parse_duration

from ._cli_utils import (
    REPO_LIST_DEFAULT_LIMIT,
    AuthorOpt,
    EnvFileOpt,
    EnvOpt,
    FilterOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    SecretsFileOpt,
    SecretsOpt,
    SshDryRunOpt,
    SshIdentityFileOpt,
    TokenOpt,
    VolumesOpt,
    exec_ssh,
    get_hf_api,
    make_expand_properties_parser,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._file_listing import list_repo_files_cmd
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


HOT_RELOADING_MIN_GRADIO = "6.1.0"


_EXPAND_PROPERTIES = sorted(get_args(ExpandSpaceProperty_T))
_SORT_OPTIONS = get_args(SpaceSort_T)
SpaceSortEnum = enum.Enum("SpaceSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]

spaces_cli = typer_factory(help="Interact with spaces on the Hub.")
volumes_cli = typer_factory(help="Manage volumes for a Space on the Hub.")
secrets_cli = typer_factory(help="Manage secrets for a Space on the Hub.")
variables_cli = typer_factory(help="Manage environment variables for a Space on the Hub.")
zero_gpu_cli = typer_factory(help="Manage your ZeroGPU usage on the Hub.")
spaces_cli.add_group(volumes_cli, name="volumes")
spaces_cli.add_group(secrets_cli, name="secrets")
spaces_cli.add_group(variables_cli, name="variables")
spaces_cli.add_group(zero_gpu_cli, name="zero-gpu")


@spaces_cli.command(
    "list | ls",
    examples=[
        "hf spaces ls --limit 10",
        'hf spaces ls --search "chatbot" --author huggingface',
        "hf spaces ls victor/deepsite",
        "hf spaces ls victor/deepsite -R",
        "hf spaces ls victor/deepsite --tree -h",
    ],
)
def spaces_ls(
    repo_id: Annotated[
        str | None,
        Argument(help="Space ID (e.g. `username/repo-name`) to list files from. If omitted, lists spaces."),
    ] = None,
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        SpaceSortEnum | None,
        Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = REPO_LIST_DEFAULT_LIMIT,
    expand: ExpandOpt = None,
    human_readable: Annotated[
        bool,
        Option("--human-readable", "-h", help="Show sizes in human readable format (only for listing files)."),
    ] = False,
    as_tree: Annotated[
        bool,
        Option("--tree", help="List files in tree format (only for listing files)."),
    ] = False,
    recursive: Annotated[
        bool,
        Option("--recursive", "-R", help="List files recursively (only for listing files)."),
    ] = False,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
) -> None:
    """List spaces on the Hub, or files in a space repo.

    When called with no argument, lists spaces on the Hub.
    When called with a space ID, lists files in that space repo.
    """
    if repo_id is not None:
        if search is not None:
            raise click.BadParameter("Cannot use --search when listing files.")
        if author is not None:
            raise click.BadParameter("Cannot use --author when listing files.")
        if filter is not None:
            raise click.BadParameter("Cannot use --filter when listing files.")
        if sort is not None:
            raise click.BadParameter("Cannot use --sort when listing files.")
        if limit != REPO_LIST_DEFAULT_LIMIT:
            raise click.BadParameter("Cannot use --limit when listing files.")
        if expand is not None:
            raise click.BadParameter("Cannot use --expand when listing files.")
        return list_repo_files_cmd(
            repo_id=repo_id,
            repo_type="space",
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            revision=revision,
            token=token,
        )

    if as_tree:
        raise click.BadParameter("Cannot use --tree when listing spaces.")
    if recursive:
        raise click.BadParameter("Cannot use --recursive when listing spaces.")
    if human_readable:
        raise click.BadParameter("Cannot use --human-readable when listing spaces.")
    if revision is not None:
        raise click.BadParameter("Cannot use --revision when listing spaces.")
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        _dataclass_to_dict(space_info)
        for space_info in api.list_spaces(
            filter=filter,
            author=author,
            search=search,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore[arg-type]
        )
    ]
    out.table(results)


@spaces_cli.command(
    "info",
    examples=[
        "hf spaces info enzostvs/deepsite",
        "hf spaces info gradio/theme_builder --expand sdk,runtime,likes",
    ],
)
def spaces_info(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a space on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.space_info(repo_id=space_id, revision=revision, expand=expand)  # type: ignore[arg-type]
    except RepositoryNotFoundError as e:
        raise CLIError(f"Space '{space_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{space_id}'.") from e
    out.dict(info)


@spaces_cli.command(
    "card",
    examples=[
        "hf spaces card mteb/leaderboard",
        "hf spaces card mteb/leaderboard --metadata",
        "hf spaces card mteb/leaderboard --metadata --format json",
        "hf spaces card mteb/leaderboard --text",
    ],
)
def spaces_card(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    metadata: Annotated[bool, Option("--metadata", help="Output only the metadata from the card.")] = False,
    text: Annotated[bool, Option("--text", help="Output only the text body (no metadata).")] = False,
    token: TokenOpt = None,
) -> None:
    """Get the Space card (README) for a Space on the Hub."""
    if metadata and text:
        raise CLIError("--metadata and --text are mutually exclusive.")
    card = SpaceCard.load(space_id, token=token)
    if metadata:
        out.dict(card.data.to_dict())
    elif text:
        out.text(card.text)
    else:
        out.text(card.content)
        out.hint(f"Use `hf spaces card {space_id} --metadata` to extract only the card metadata.")


@spaces_cli.command(
    "templates",
    examples=["hf spaces templates"],
)
def spaces_templates(
    token: TokenOpt = None,
) -> None:
    """List the available Space templates.

    The `repo_id` (or `name`) of a template can be passed to `hf repos create --template ...` to
    create a new Space from that template.
    """
    api = get_hf_api(token=token)
    templates = [_dataclass_to_dict(template) for template in api.list_space_templates()]
    out.table(templates, id_key="name")
    out.hint(
        "Create a Space from a template with `hf repos create <id> --type space --space-sdk <sdk> --template <repo_id>`."
    )


@spaces_cli.command(
    "search",
    examples=[
        'hf spaces search "generate image"',
        'hf spaces search "identify objects in pictures" --sdk gradio --limit 5',
        'hf spaces search "remove background from photo" --description --json',
    ],
)
def spaces_search(
    query: Annotated[str, Argument(help="Search query.")],
    filter: FilterOpt = None,
    sdk: Annotated[list[str] | None, Option(help="Filter by SDK (e.g. gradio, docker, static).")] = None,
    include_non_running: Annotated[bool, Option(help="Include non-running spaces in results.")] = False,
    description: Annotated[bool, Option(help="Show AI-generated descriptions.")] = False,
    limit: LimitOpt = 10,
    token: TokenOpt = None,
) -> None:
    """Search spaces on the Hub using semantic search."""
    api = get_hf_api(token=token)
    results = api.search_spaces(
        query=query,
        filter=filter,
        sdk=sdk,
        include_non_running=include_non_running,
        token=token,
    )
    items = []
    for r in itertools.islice(results, limit):
        item: dict = {
            "id": r.id,
            "title": r.title,
            "sdk": r.sdk,
            "likes": r.likes,
            "stage": r.runtime.stage if r.runtime else None,
            "category": r.ai_category,
            "score": round(r.semantic_relevancy_score, 2) if r.semantic_relevancy_score is not None else None,
        }
        if description:
            item["description"] = r.ai_short_description
        items.append(item)
    out.table(items)
    if not description:
        out.hint("Use --description to show AI-generated descriptions.")


@spaces_cli.command(
    "wait",
    examples=[
        "hf spaces wait username/my-space",
        "hf spaces wait username/my-space --timeout 5m",
    ],
)
def spaces_wait(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    timeout: Annotated[
        str | None,
        Option(
            help="Max time to wait: int with s (seconds, default), m (minutes), h (hours) or d (days).",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Wait for a Space to finish building/starting.

    Blocks until the Space leaves an intermediate stage (BUILDING, APP_STARTING, etc.)
    and reaches a settled stage. Exits with code 0 if the Space is RUNNING,
    or a non-zero exit code otherwise (e.g. BUILD_ERROR, RUNTIME_ERROR).
    """
    timeout_secs = parse_duration(timeout) if timeout is not None else None
    api = get_hf_api(token=token)
    status = out.status("Waiting for Space to be ready...")
    try:
        runtime = api.wait_for_space(space_id, timeout=timeout_secs)
    except TimeoutError:
        status.done("Timed out.")
        raise CLIError(f"Timed out after {timeout} waiting for Space '{space_id}' to be ready.") from None
    status.done(f"Space reached stage '{runtime.stage}'.")
    if runtime.stage != SpaceStage.RUNNING:
        raise CLIError(f"Space '{space_id}' is not running (stage='{runtime.stage}').")
    out.result("Space ready", space_id=space_id, stage=str(runtime.stage))
    out.hint(f"Use `hf spaces logs {space_id}` to view run logs.")


@spaces_cli.command(
    "dev-mode",
    examples=[
        "hf spaces dev-mode my-user-name/deepsite",
    ],
)
def dev_mode(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    stop: Annotated[bool, Option(help="Stop dev mode.")] = False,
    token: TokenOpt = None,
):
    """
    Enable or disable dev mode on a Space.

    Spaces Dev Mode eases the debugging of your application and makes iterating on Spaces faster by allowing you to
    restart your application without stopping the Space container itself. Enabling dev mode requires a PRO or Team &
    Enterprise plan. Disabling it (`--stop`) is always allowed.

    See docs: https://huggingface.co/docs/hub/spaces-dev-mode
    """
    api = get_hf_api(token=token)
    if stop:
        api.disable_space_dev_mode(space_id)
        print(f"Dev mode disabled for '{space_id}'")
        return
    api.enable_space_dev_mode(space_id)
    runtime = api.wait_for_space(space_id)
    if runtime.stage != SpaceStage.RUNNING:
        out.warning(f"Dev mode is not ready (stage='{runtime.stage}')")
        return
    info = api.space_info(space_id)
    folder = getattr(info.card_data, "dev-mode-folder", "" if info.sdk == "docker" else "/home/user/app")
    folder_query_param = f"folder={folder}" if folder else ""
    print("Connect to dev environment:")
    print("")
    print("Web:")
    vscode_web_url = f"https://huggingface.co/spaces/{info.id}/dev-mode/vscode-web"
    if folder_query_param:
        vscode_web_url += f"?{folder_query_param}"
    ssh_host = f"{info.subdomain}@ssh.hf.space"
    print(f"  * VSCode: {vscode_web_url}")
    print("")
    print("Local:")
    print("1. Add your SSH key to https://huggingface.co/settings/keys")
    print(f"2. SSH with `hf spaces ssh {space_id}` (or `ssh -i <your_key> {ssh_host}`)")
    print("   Or open")
    print(f"  * VSCode: vscode://vscode-remote/ssh-remote+{ssh_host}{folder}")
    print(f"  * Cursor: cursor://vscode-remote/ssh-remote+{ssh_host}{folder}")
    print("")
    print("PS: Dev mode stops after 48h of inactivity, don't forget to save your changes regularly.")


@spaces_cli.command(
    "ssh",
    examples=[
        "hf spaces ssh username/my-space",
        "hf spaces ssh username/my-space --dry-run",
        "hf spaces ssh username/my-space -i ~/.ssh/id_ed25519",
        "hf spaces ssh username/my-space --auto",
    ],
)
def spaces_ssh(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    identity_file: SshIdentityFileOpt = None,
    dry_run: SshDryRunOpt = False,
    auto: Annotated[
        bool,
        Option("--auto", help="Enable Dev Mode without prompting if not already enabled."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """SSH into a Space's Dev Mode container.

    Requires Dev Mode to be running on the Space and your SSH public key to be registered at https://huggingface.co/settings/keys.

    See: https://huggingface.co/docs/hub/spaces-dev-mode
    """
    api = get_hf_api(token=token)
    info = api.space_info(space_id)
    if info.runtime is None or not info.runtime.dev_mode:
        out.confirm(
            f"Dev Mode is disabled on '{space_id}'. Enable it now?", yes=auto, default=True, confirm_param="--auto"
        )
        api.enable_space_dev_mode(space_id)
        runtime = api.wait_for_space(space_id)
        if runtime.stage != SpaceStage.RUNNING:
            raise CLIError(f"Space '{space_id}' is not running (stage='{runtime.stage}').")
        info = api.space_info(space_id)
    exec_ssh(f"{info.subdomain}@ssh.hf.space", identity_file=identity_file, dry_run=dry_run)


@spaces_cli.command(
    "pause",
    examples=[
        "hf spaces pause username/my-space",
    ],
)
def spaces_pause(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    token: TokenOpt = None,
) -> None:
    """Pause a Space."""
    api = get_hf_api(token=token)
    runtime = api.pause_space(space_id)
    out.result("Space paused", space_id=space_id, stage=runtime.stage)
    out.hint(f"Use `hf spaces restart {space_id}` to restart it.")
    out.hint(
        f"Mount a Volume or bucket to persist data across restarts: `hf spaces volumes set {space_id} -v hf://...`"
    )


@spaces_cli.command(
    "restart",
    examples=[
        "hf spaces restart username/my-space",
        "hf spaces restart username/my-space --factory-reboot",
    ],
)
def spaces_restart(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    factory_reboot: Annotated[
        bool,
        Option(
            "--factory-reboot",
            help="Rebuild the Space from scratch without using the build cache.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Restart a Space."""
    api = get_hf_api(token=token)
    runtime = api.restart_space(space_id, factory_reboot=factory_reboot)
    out.result(
        "Space restart triggered",
        space_id=space_id,
        stage=runtime.stage,
        factory_reboot=factory_reboot,
    )
    out.hint(f"Use `hf spaces wait {space_id}` to wait until the Space is ready.")
    out.hint(
        f"Mount a Volume or bucket to persist data across restarts: `hf spaces volumes set {space_id} -v hf://...`"
    )


@spaces_cli.command(
    "hardware",
    examples=[
        "hf spaces hardware",
    ],
)
def spaces_hardware(token: TokenOpt = None) -> None:
    """List available hardware options for Spaces."""
    api = get_hf_api(token=token)
    hardware_list = api.list_spaces_hardware()
    items = []
    for hw in hardware_list:
        accelerator = (
            f"{hw.accelerator.quantity}x {hw.accelerator.model} ({hw.accelerator.vram})" if hw.accelerator else None
        )
        cost_min = f"${hw.unit_cost_usd:.4f}" if hw.unit_cost_usd else "free"
        cost_hour = f"${hw.unit_cost_usd * 60:.2f}" if hw.unit_cost_usd else "free"
        items.append(
            {
                "name": hw.name,
                "pretty name": hw.pretty_name,
                "cpu": hw.cpu,
                "ram": hw.ram,
                "accelerator": accelerator,
                "cost/min": cost_min,
                "cost/hour": cost_hour,
            }
        )
    out.table(items)
    out.hint("Use `hf spaces settings <space_id> --hardware <name>` to request hardware for a Space.")


@zero_gpu_cli.command(
    "quota",
    examples=[
        "hf spaces zero-gpu quota",
        "hf spaces zero-gpu quota --format json",
    ],
)
def spaces_zero_gpu_quota(token: TokenOpt = None) -> None:
    """Show your ZeroGPU quota (remaining GPU time and reset date).

    Values are in GPU-seconds. With `--quiet`, only the remaining GPU-seconds are printed.
    """
    api = get_hf_api(token=token)
    quota = api.get_zero_gpu_quota()
    out.result(
        "ZeroGPU quota (in GPU-seconds)",
        remaining=quota.remaining,
        base=quota.base,
        resets_at=quota.resets_at.isoformat() if quota.resets_at else None,
        overquota_used=quota.overquota_used,
    )
    if quota.remaining < 0.1 * quota.base:
        status = "exhausted" if quota.remaining <= 0 else "running low"
        out.hint(
            f"Your ZeroGPU quota is {status}. PRO, Team and Enterprise users can purchase credits to keep using"
            " ZeroGPU beyond the daily quota: https://huggingface.co/settings/billing?add-credits=true"
        )


@spaces_cli.command(
    "settings",
    examples=[
        "hf spaces settings username/my-space --sleep-time 300",
        "hf spaces settings username/my-space --hardware t4-medium",
    ],
)
def spaces_settings(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    sleep_time: Annotated[
        int | None,
        Option(
            "--sleep-time",
            help="Idle time in seconds after which the Space goes to sleep. Use -1 to never sleep. Only available on upgraded hardware.",
        ),
    ] = None,
    hardware: Annotated[
        str | None,
        Option(
            "--hardware",
            help="Space hardware flavor (e.g. 'cpu-basic', 't4-medium', 'l4x4'). Run 'hf spaces hardware' to list available options.",
            click_type=SoftChoice(SpaceHardware),
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update the settings of a Space."""
    api = get_hf_api(token=token)
    if hardware is not None:
        runtime = api.request_space_hardware(space_id, hardware=hardware, sleep_time=sleep_time)  # type: ignore[arg-type]
    elif sleep_time is not None:
        runtime = api.set_space_sleep_time(space_id, sleep_time=sleep_time)
    else:
        raise CLIError("Specify at least one setting to update.")
    out.result(
        "Space settings updated",
        space_id=space_id,
        hardware=runtime.requested_hardware,
        sleep_time=runtime.sleep_time,
    )
    out.hint(f"Use `hf spaces info {space_id}` to verify the runtime configuration.")


@spaces_cli.command(
    "logs",
    examples=[
        "hf spaces logs username/my-space",
        "hf spaces logs username/my-space --build",
        "hf spaces logs -f username/my-space",
        "hf spaces logs -n 50 username/my-space",
    ],
)
def spaces_logs(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    build: Annotated[
        bool,
        Option(
            "--build",
            help="Fetch the container build logs instead of the run logs. Useful when a Space is stuck in BUILD_ERROR.",
        ),
    ] = False,
    follow: Annotated[
        bool,
        Option(
            "-f",
            "--follow",
            help="Follow log output (stream until the server closes the stream). Without this flag, only currently available logs are printed.",
        ),
    ] = False,
    tail: Annotated[
        int | None,
        Option(
            "-n",
            "--tail",
            help="Number of lines to show from the end of the logs.",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the run or build logs of a Space.

    By default, prints currently available run logs and exits (non-blocking, like
    `docker logs`). Use --follow/-f to stream until the server closes the stream.
    Use --build to see the container build logs instead (useful when a Space is
    stuck in BUILD_ERROR).
    """
    if follow and tail is not None:
        raise CLIError(
            "Cannot use --follow and --tail together. Use --follow to stream logs or --tail to show recent logs."
        )

    api = get_hf_api(token=token)
    logs = api.fetch_space_logs(space_id, build=build, follow=follow)
    if tail is not None:
        logs = deque(logs, maxlen=tail)
    found_logs = False
    for line in logs:
        clean_line = line.strip()
        out.text(clean_line)
        if clean_line:
            found_logs = True
    if not found_logs and not build:
        out.hint(f"No run logs found for space {space_id}. Try passing --build to fetch build logs instead.")


@spaces_cli.command(
    "hot-reload",
    examples=[
        "hf spaces hot-reload username/repo-name app.py     # Open an interactive editor to the remote app.py file",
        "hf spaces hot-reload username/repo-name -f app.py  # Take local version from ./app.py and patch app.py remotely",
        "hf spaces hot-reload username/repo-name app.py -f src/app.py # Take local version from ./src/app.py",
    ],
)
def spaces_hot_reload(
    space_id: Annotated[
        str,
        Argument(
            help="The space ID (e.g. `username/repo-name`).",
        ),
    ],
    filename: Annotated[
        str | None,
        Argument(
            help="Path to the Python file in the Space repository. Can be omitted when --local-file is specified and path in repository matches."
        ),
    ] = None,
    local_file: Annotated[
        Path | None,
        Option(
            "--local-file",
            "-f",
            help="Path of local file. Interactive editor mode if not specified",
        ),
    ] = None,
    skip_checks: Annotated[bool, Option(help="Skip hot-reload compatibility checks.")] = False,
    skip_summary: Annotated[bool, Option(help="Skip summary display after hot-reload is triggered")] = False,
    token: TokenOpt = None,
) -> None:
    """
    Hot-reload any Python file of a Space without a full rebuild + restart.

    ⚠ This feature is experimental ⚠

    Only works with Gradio SDK (6.1+)
    Opens an interactive editor unless --local-file/-f is specified.

    This command patches the live Python process using https://github.com/breuleux/jurigged
    (AST-based diffing, in-place function updates, etc.), integrated with Gradio's native hot-reload support
    (meaning that Gradio demo object changes are reflected in the UI)

    The command creates a remote commit.
    If you are working from a local clone, run `git pull --autostash` afterwards
    to bring the commit back and keep your local git state in sync.
    """

    click.secho("This feature is experimental and subject to change", fg="bright_black")

    api = get_hf_api(token=token)

    if not skip_checks:
        space_info = api.space_info(space_id)
        if space_info.sdk != "gradio":
            raise CLIError(f"Hot-reloading is only available on Gradio SDK. Found {space_info.sdk} SDK")
        if (card_data := space_info.card_data) is None:
            raise CLIError(f"Unable to read cardData for Space {space_id}")
        if (sdk_version := card_data.sdk_version) is None:
            raise CLIError(f"Unable to read sdk_version from {space_id} cardData")
        if version.parse(sdk_version) < version.Version(HOT_RELOADING_MIN_GRADIO):
            raise CLIError(f"Hot-reloading requires Gradio >= {HOT_RELOADING_MIN_GRADIO} (found {sdk_version})")
        if (current_sha := space_info.sha) is None:
            raise CLIError(f"Unexpected `None` running SHA for Space {space_id}")
    else:
        current_sha = None

    if local_file:
        local_path = str(local_file)
        filename = local_file.as_posix() if filename is None else filename
    elif filename:
        if not skip_checks:
            try:
                api.auth_check(
                    repo_type="space",
                    repo_id=space_id,
                    write=True,
                )
            except RepositoryNotFoundError as e:
                raise CLIError(
                    f"Write access check to {space_id} repository failed. Make sure that you are authenticated"
                ) from e
        temp_dir = tempfile.TemporaryDirectory()
        local_path = os.path.join(temp_dir.name, filename)
        with disable_progress_bars():
            try:
                hf_hub_download(repo_type="space", repo_id=space_id, filename=filename, local_dir=temp_dir.name)
            except RemoteEntryNotFoundError:
                click.secho(f"{filename} not found in remote repository. Assuming new file", fg="bright_black")

        editor_res = _editor_open(local_path)
        if editor_res == "no-tty":
            persistent_temp_dir = tempfile.mkdtemp()
            shutil.copytree(temp_dir.name, persistent_temp_dir, dirs_exist_ok=True)
            local_path = os.path.join(persistent_temp_dir, filename)
            click.secho("No TTY detected. Non-interactive fallback:")
            click.secho(f"- Edit {local_path}")
            click.secho(f"- Run `hf spaces hot-reload {space_id} {filename} -f {local_path}`")
            return
        if editor_res == "no-editor":
            raise CLIError("No editor found in local environment. Use -f flag to hot-reload from local path")
        if editor_res != 0:
            raise CLIError(f"Editor returned a non-zero exit code while attempting to edit {local_path}")
    else:
        raise CLIError("Either filename or --local-file/-f must be specified")

    commit_info = api.upload_file(
        repo_type="space",
        repo_id=space_id,
        path_or_fileobj=local_path,
        path_in_repo=filename,
        parent_commit=current_sha,
        _hot_reload=True,
    )

    if local_file is not None and local_file.resolve().is_relative_to(Path.cwd()):
        click.secho(f"Created commit {commit_info.oid} in remote Space repository.")
        click.secho("Consider running `git pull --autostash` to stay synced if you are working from a local clone.")

    if not skip_summary:
        click.secho("Hot-reload summary:")
        _spaces_hot_reload_summary(
            api=api,
            space_id=space_id,
            current_sha=current_sha,
            commit_sha=commit_info.oid,
            local_path=local_path if local_file else filename,
            filename=filename,
            token=token,
        )


def _spaces_hot_reload_summary(
    api: HfApi,
    space_id: str,
    current_sha: str | None,
    commit_sha: str,
    filename: str,
    local_path: str,
    token: str | None,
) -> None:
    while (space_info := api.space_info(space_id)).sha == current_sha:
        if current_sha is None or current_sha == commit_sha:
            break
        click.secho("Waiting for up-to-date Space infos", fg="bright_black", err=True)
        time.sleep(2)
    if space_info.sha != commit_sha:
        raise CLIError(f"Expected SHA {commit_sha} after hot-reload but got {space_info.sha}")
    if (runtime := space_info.runtime) is None:
        raise CLIError(f"Unable to read SpaceRuntime from {space_id} infos")
    if (hot_reloading := runtime.hot_reloading) is None:
        raise CLIError(f"Space {space_id} current running version has not been hot-reloaded")
    if hot_reloading.status != "created":
        click.echo(f"Failed creating hot-reloaded commit. {hot_reloading.replica_statuses=}")
        return

    if (space_host := space_info.host) is None:
        raise CLIError("Unexpected None host on hotReloaded Space")
    if (space_subdomain := space_info.subdomain) is None:
        raise CLIError("Unexpected None subdomain on hotReloaded Space")

    def render_region(region: ReloadRegion) -> str:
        res = f"{local_path}, "
        if region["startLine"] == region["endLine"]:
            res += f"line {region['startLine'] - 1}"
        else:
            res += f"lines {region['startLine'] - 1}-{region['endLine']}"
        return res

    def display_event(event: ApiGetReloadEventSourceData) -> None:
        if event["data"]["kind"] == "error":
            click.secho("✘ Unexpected hot-reloading error", bold=True)
            click.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "exception":
            click.secho(f"✘ Exception at {render_region(event['data']['region'])}", bold=True)
            click.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "add":
            click.secho(f"✔︎ Created {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "delete":
            click.secho(f"∅ Deleted {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "update":
            click.secho(f"✔︎ Updated {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "run":
            click.secho(f"▶ Run {render_region(event['data']['region'])}", bold=True)
            click.secho(event["data"]["codeLines"], italic=True)
        elif event["data"]["kind"] == "ui":
            if event["data"]["updated"]:
                click.secho("⟳ UI updated", bold=True)
            else:
                click.secho("∅ UI untouched", bold=True)
        elif event["data"]["kind"] == "file":
            if event["data"]["created"]:
                click.secho(f"✔︎ {filename} created", bold=True)
            else:
                click.secho(f"✔︎ {filename} updated", bold=True)
        else:
            click.secho(f"❓ Unknown update event: {event=}")
            if TYPE_CHECKING:
                assert_never(event["data"]["kind"])

    for replica_stream_event in multi_replica_reload_events(
        commit_sha=commit_sha,
        host=space_host,
        subdomain=space_subdomain,
        replica_hashes=[hash for hash, _ in hot_reloading.replica_statuses],
        token=token,
    ):
        if replica_stream_event["kind"] == "event":
            display_event(replica_stream_event["event"])
        elif replica_stream_event["kind"] == "replicaHash":
            click.secho(f"---- Replica {replica_stream_event['hash']} ----")
        elif replica_stream_event["kind"] == "fullMatch":
            click.echo("✔︎ Same as first replica")
        elif replica_stream_event["kind"] == "warning":
            click.secho(f"⚠ {replica_stream_event['message']}", fg="bright_black")
        else:
            assert_never(replica_stream_event)


PREFERRED_EDITORS = (
    ("code", "code --wait"),
    ("nvim", "nvim"),
    ("nano", "nano"),
    ("vim", "vim"),
    ("vi", "vi"),
)


@functools.cache
def _get_editor_command() -> str | None:
    for env in ("HF_EDITOR", "VISUAL", "EDITOR"):
        if command := os.getenv(env, "").strip():
            return command
    for binary_path, editor_command in PREFERRED_EDITORS:
        if shutil.which(binary_path) is not None:
            return editor_command
    return None


def _editor_open(local_path: str) -> int | Literal["no-tty", "no-editor"]:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "no-tty"
    if (editor_command := _get_editor_command()) is None:
        return "no-editor"
    command = [*shlex.split(editor_command), local_path]
    res = subprocess.run(command, start_new_session=True)
    return res.returncode


@volumes_cli.command(
    "list | ls",
    examples=[
        "hf spaces volumes ls username/my-space",
    ],
)
def volumes_ls(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    token: TokenOpt = None,
) -> None:
    """List volumes mounted in a Space."""
    api = get_hf_api(token=token)
    info = api.space_info(space_id)
    if info.runtime is None:
        raise CLIError(f"Runtime not available for Space '{space_id}'.")
    volumes = info.runtime.volumes or []
    items = [_dataclass_to_dict(v) for v in volumes]
    out.table(items)
    out.hint(
        f"Use `hf spaces volumes set {space_id} -v hf://<repo_type>/<repo_id>:/<mount_path>` to set volumes for a Space."
    )


@volumes_cli.command(
    "set",
    examples=[
        "hf spaces volumes set username/my-space -v hf://models/username/my-model:/models",
        "hf spaces volumes set username/my-space -v hf://buckets/username/my-bucket:/data -v hf://datasets/username/my-dataset:/datasets:ro",
    ],
)
def volumes_set(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    volume: VolumesOpt = None,
    token: TokenOpt = None,
) -> None:
    """Set (replace) volumes for a Space."""
    volumes = parse_volumes(volume)
    if not volumes:
        raise CLIError("At least one volume must be specified with -v/--volume.")
    api = get_hf_api(token=token)
    api.set_space_volumes(space_id, volumes=volumes)
    out.result("Volumes set", space_id=space_id, volumes=[v.to_uri() for v in volumes])
    out.hint(f"Use `hf spaces volumes ls {space_id}` to list volumes for a Space.")


@volumes_cli.command(
    "delete",
    examples=[
        "hf spaces volumes delete username/my-space",
        "hf spaces volumes delete username/my-space --yes",
    ],
)
def volumes_delete(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Remove all volumes from a Space."""
    out.confirm(f"You are about to remove all volumes from Space '{space_id}'. Proceed?", yes=yes)
    api = get_hf_api(token=token)
    api.delete_space_volumes(space_id)
    out.result("Volumes deleted", space_id=space_id)
    out.hint(
        f"Use `hf spaces volumes set {space_id} -v hf://<repo_type>/<repo_id>:/<mount_path>` to set volumes for a Space."
    )


@secrets_cli.command(
    "list | ls",
    examples=["hf spaces secrets ls username/my-space"],
)
def secrets_ls(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    token: TokenOpt = None,
) -> None:
    """List secrets for a Space. Secret values are write-only and not returned."""
    api = get_hf_api(token=token)
    secrets = api.get_space_secrets(space_id)
    items = [_dataclass_to_dict(s) for s in secrets.values()]
    out.table(items)
    out.hint(f"Use `hf spaces secrets add {space_id} -s KEY=VALUE` to add secrets to a Space.")


@secrets_cli.command(
    "add",
    examples=[
        "hf spaces secrets add username/my-space -s HF_TOKEN",
        "hf spaces secrets add username/my-space -s OPENAI_API_KEY=sk-... -s ANTHROPIC_API_KEY=sk-...",
        "hf spaces secrets add username/my-space --secrets-file .env.secrets",
    ],
)
def secrets_add(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    secrets: SecretsOpt = None,
    secrets_file: SecretsFileOpt = None,
    token: TokenOpt = None,
) -> None:
    """Add or update secrets for a Space."""
    secrets_map = parse_env_map(secrets, secrets_file)
    if not secrets_map:
        raise CLIError("At least one secret must be specified with -s/--secrets or --secrets-file.")
    api = get_hf_api(token=token)
    for key, value in secrets_map.items():
        api.add_space_secret(space_id, key=key, value=value or "")
    out.result("Secrets added", space_id=space_id, keys=list(secrets_map))
    out.hint(f"Use `hf spaces secrets delete {space_id} <key>` to remove a secret from a Space.")


@secrets_cli.command(
    "delete",
    examples=[
        "hf spaces secrets delete username/my-space HF_TOKEN",
        "hf spaces secrets delete username/my-space HF_TOKEN --yes",
    ],
)
def secrets_delete(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    key: Annotated[str, Argument(help="Name of the secret to remove.")],
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Remove a secret from a Space."""
    out.confirm(
        f"You are about to remove secret '{key}' from Space '{space_id}'. The value cannot be recovered. Proceed?",
        yes=yes,
    )
    api = get_hf_api(token=token)
    api.delete_space_secret(space_id, key=key)
    out.result("Secret deleted", space_id=space_id, key=key)
    out.hint(f"Use `hf spaces secrets add {space_id} -s {key}=<value>` to re-add a secret to a Space.")


@variables_cli.command(
    "list | ls",
    examples=["hf spaces variables ls username/my-space"],
)
def variables_ls(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    token: TokenOpt = None,
) -> None:
    """List environment variables for a Space."""
    api = get_hf_api(token=token)
    variables = api.get_space_variables(space_id)
    items = [_dataclass_to_dict(v) for v in variables.values()]
    out.table(items)
    out.hint(f"Use `hf spaces variables add {space_id} -e KEY=VALUE` to add variables to a Space.")


@variables_cli.command(
    "add",
    examples=[
        "hf spaces variables add username/my-space -e DEBUG=1",
        "hf spaces variables add username/my-space -e MODEL_ID=gpt2 -e MAX_TOKENS=512",
        "hf spaces variables add username/my-space --env-file .env",
    ],
)
def variables_add(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    token: TokenOpt = None,
) -> None:
    """Add or update environment variables for a Space."""
    env_map = parse_env_map(env, env_file)
    if not env_map:
        raise CLIError("At least one variable must be specified with -e/--env or --env-file.")
    api = get_hf_api(token=token)
    for key, value in env_map.items():
        api.add_space_variable(space_id, key=key, value=value or "")
    out.result("Variables added", space_id=space_id, keys=list(env_map))
    out.hint(f"Use `hf spaces variables ls {space_id}` to list variables for a Space.")


@variables_cli.command(
    "delete",
    examples=[
        "hf spaces variables delete username/my-space DEBUG",
        "hf spaces variables delete username/my-space DEBUG --yes",
    ],
)
def variables_delete(
    space_id: Annotated[str, Argument(help="The space ID (e.g. `username/repo-name`).")],
    key: Annotated[str, Argument(help="Name of the variable to remove.")],
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Remove an environment variable from a Space."""
    out.confirm(
        f"You are about to remove variable '{key}' from Space '{space_id}'. Proceed?",
        yes=yes,
    )
    api = get_hf_api(token=token)
    api.delete_space_variable(space_id, key=key)
    out.result("Variable deleted", space_id=space_id, key=key)
    out.hint(f"Use `hf spaces variables ls {space_id}` to list remaining variables for a Space.")
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
"""Contains commands to interact with buckets via the CLI."""

from collections.abc import Iterable, Iterator
from typing import Annotated

import click

from huggingface_hub import logging
from huggingface_hub._buckets import (
    BUCKET_PREFIX,
    BucketFile,
    BucketFolder,
    FilterMatcher,
    _parse_bucket_uri,
)

from ..hf_api import REPO_REGIONS
from ._cli_utils import (
    SearchOpt,
    TokenOpt,
    get_hf_api,
    typer_factory,
)
from ._cp import make_cp
from ._file_listing import format_size, print_file_listing
from ._framework import Argument, Option
from ._output import OutputFormat, out


logger = logging.get_logger(__name__)


buckets_cli = typer_factory(help="Commands to interact with buckets.")


@buckets_cli.command(
    name="create",
    examples=[
        "hf buckets create my-bucket",
        "hf buckets create user/my-bucket",
        "hf buckets create hf://buckets/user/my-bucket",
        "hf buckets create user/my-bucket --private",
        "hf buckets create user/my-bucket --exist-ok",
        "hf buckets create user/my-bucket --region us",
    ],
)
def create(
    bucket_id: Annotated[
        str,
        Argument(
            help="Bucket ID: bucket_name, namespace/bucket_name, or hf://buckets/namespace/bucket_name",
        ),
    ],
    private: Annotated[
        bool,
        Option(
            "--private",
            help="Create a private bucket.",
        ),
    ] = False,
    region: Annotated[
        REPO_REGIONS | None,
        Option(
            "--region",
            help="Cloud region in which to create the bucket. Can be one of 'us' or 'eu'. Requires Team plan or above.",
        ),
    ] = None,
    exist_ok: Annotated[
        bool,
        Option(
            "--exist-ok",
            help="Do not raise an error if the bucket already exists.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Create a new bucket."""
    api = get_hf_api(token=token)

    if bucket_id.startswith(BUCKET_PREFIX):
        parsed = _parse_bucket_uri(bucket_id)
        if parsed.path_in_repo:
            raise click.BadParameter(
                f"Cannot specify a prefix for bucket creation: {bucket_id}."
                f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
            )
        bucket_id = parsed.id

    bucket_url = api.create_bucket(
        bucket_id,
        private=private if private else None,
        region=region,
        exist_ok=exist_ok,
    )
    out.result("Bucket created", uri=bucket_url.uri.to_uri(), url=bucket_url.url)


def _is_bucket_id(argument: str) -> bool:
    """Check if argument is a bucket ID (namespace/name) vs just a namespace."""
    if argument.startswith(BUCKET_PREFIX):
        path = argument[len(BUCKET_PREFIX) :]
    else:
        path = argument
    return "/" in path


@buckets_cli.command(
    name="list | ls",
    examples=[
        "hf buckets list",
        "hf buckets list huggingface",
        'hf buckets list --search "my-prefix"',
        "hf buckets list user/my-bucket",
        "hf buckets list user/my-bucket -R",
        "hf buckets list user/my-bucket -h",
        "hf buckets list user/my-bucket --tree",
        "hf buckets list user/my-bucket --tree -h",
        "hf buckets list hf://buckets/user/my-bucket",
        "hf buckets list user/my-bucket/sub -R",
    ],
)
def list_cmd(
    argument: Annotated[
        str | None,
        Argument(
            help=(
                "Namespace (user or org) to list buckets, or bucket ID"
                " (namespace/bucket_name(/prefix) or hf://buckets/...) to list files."
            ),
        ),
    ] = None,
    human_readable: Annotated[
        bool,
        Option(
            "--human-readable",
            "-h",
            help="Show sizes in human readable format.",
        ),
    ] = False,
    as_tree: Annotated[
        bool,
        Option(
            "--tree",
            help="List files in tree format (only for listing files).",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        Option(
            "--recursive",
            "-R",
            help="List files recursively (only for listing files).",
        ),
    ] = False,
    search: SearchOpt = None,
    token: TokenOpt = None,
) -> None:
    """List buckets or files in a bucket.

    When called with no argument or a namespace, lists buckets.
    When called with a bucket ID (namespace/bucket_name), lists files in the bucket.
    """
    # Determine mode: listing buckets or listing files
    is_file_mode = argument is not None and _is_bucket_id(argument)

    if is_file_mode:
        if search is not None:
            raise click.BadParameter("Cannot use --search when listing files.")
        _list_files(
            argument=argument,  # type: ignore
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            token=token,
        )
    else:
        _list_buckets(
            namespace=argument,
            search=search,
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            token=token,
        )


def _list_buckets(
    namespace: str | None,
    search: str | None,
    human_readable: bool,
    as_tree: bool,
    recursive: bool,
    token: str | None,
) -> None:
    """List buckets in a namespace."""
    # Validate incompatible flags
    if as_tree:
        raise click.BadParameter("Cannot use --tree when listing buckets.")
    if recursive:
        raise click.BadParameter("Cannot use --recursive when listing buckets.")

    # Handle hf://buckets/namespace format
    if namespace is not None and namespace.startswith(BUCKET_PREFIX):
        namespace = namespace[len(BUCKET_PREFIX) :]
        # Strip trailing slash if any
        namespace = namespace.rstrip("/")

    api = get_hf_api(token=token)
    items = [
        {
            "id": bucket.id,
            "private": bucket.private,
            "size": format_size(bucket.size, human_readable) if human_readable else bucket.size,
            "total_files": bucket.total_files,
            "created_at": bucket.created_at,
        }
        for bucket in api.list_buckets(namespace=namespace, search=search)
    ]
    out.table(items, alignments={"size": "right"})


def _iter_under_prefix(items: Iterable[BucketFile | BucketFolder], prefix: str) -> Iterator[BucketFile | BucketFolder]:
    """Drop entries that are not `prefix` itself or below it.

    The server matches `prefix` lexically, so listing "logs" also returns "logs.json" or "logs_backup/...".
    Filesystem semantics require path components => keep only what's actually under the prefix.
    """
    prefix = prefix.rstrip("/")
    for item in items:
        if not prefix or item.path == prefix or item.path.startswith(f"{prefix}/"):
            yield item


def _list_files(
    argument: str,
    human_readable: bool,
    as_tree: bool,
    recursive: bool,
    token: str | None,
) -> None:
    """List files in a bucket."""
    if as_tree and out.mode == OutputFormat.json:
        raise click.BadParameter("Cannot use --tree with --format json.")

    api = get_hf_api(token=token)
    parsed = _parse_bucket_uri(argument)
    items = list(
        _iter_under_prefix(
            api.list_bucket_tree(
                parsed.id,
                prefix=parsed.path_in_repo or None,
                recursive=recursive,
            ),
            parsed.path_in_repo,
        )
    )

    print_file_listing(items, human_readable=human_readable, as_tree=as_tree, recursive=recursive)


@buckets_cli.command(
    name="info",
    examples=[
        "hf buckets info user/my-bucket",
        "hf buckets info hf://buckets/user/my-bucket",
    ],
)
def info(
    bucket_id: Annotated[
        str,
        Argument(
            help="Bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    token: TokenOpt = None,
) -> None:
    """Get info about a bucket."""
    api = get_hf_api(token=token)
    parsed = _parse_bucket_uri(bucket_id)
    bucket = api.bucket_info(parsed.id)
    out.dict(bucket, id_key="id")


@buckets_cli.command(
    name="delete",
    examples=[
        "hf buckets delete user/my-bucket",
        "hf buckets delete hf://buckets/user/my-bucket",
        "hf buckets delete user/my-bucket --yes",
        "hf buckets delete user/my-bucket --missing-ok",
    ],
)
def delete(
    bucket_id: Annotated[
        str,
        Argument(
            help="Bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    missing_ok: Annotated[
        bool,
        Option(
            "--missing-ok",
            help="Do not raise an error if the bucket does not exist.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a bucket.

    This deletes the entire bucket and all its contents. Use `hf buckets rm` to remove individual files.
    """
    if bucket_id.startswith(BUCKET_PREFIX):
        parsed = _parse_bucket_uri(bucket_id)
        if parsed.path_in_repo:
            raise click.BadParameter(
                f"Cannot specify a prefix for bucket deletion: {bucket_id}."
                f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
            )
        bucket_id = parsed.id
    elif "/" not in bucket_id:
        raise click.BadParameter(
            f"Invalid bucket ID: {bucket_id}."
            f" Must be in format namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
        )

    out.confirm(f"Are you sure you want to delete bucket '{bucket_id}'?", yes=yes)

    api = get_hf_api(token=token)
    api.delete_bucket(bucket_id, missing_ok=missing_ok)
    out.result("Bucket deleted", bucket_id=bucket_id)


@buckets_cli.command(
    name="remove | rm",
    examples=[
        "hf buckets remove user/my-bucket/file.txt",
        "hf buckets rm hf://buckets/user/my-bucket/file.txt",
        "hf buckets rm user/my-bucket/logs/ --recursive",
        'hf buckets rm user/my-bucket --recursive --include "*.tmp"',
        "hf buckets rm user/my-bucket/data/ --recursive --dry-run",
    ],
)
def remove(
    argument: Annotated[
        str,
        Argument(
            help=(
                "Bucket path: namespace/bucket_name/path or hf://buckets/namespace/bucket_name/path."
                " With --recursive, namespace/bucket_name is also accepted to target all files."
            ),
        ),
    ],
    recursive: Annotated[
        bool,
        Option(
            "--recursive",
            "-R",
            help="Remove files recursively under the given prefix.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            "--dry-run",
            help="Preview what would be deleted without actually deleting.",
        ),
    ] = False,
    include: Annotated[
        list[str] | None,
        Option(
            help="Include only files matching pattern (can specify multiple). Requires --recursive.",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option(
            help="Exclude files matching pattern (can specify multiple). Requires --recursive.",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Remove files from a bucket.

    To delete an entire bucket, use `hf buckets delete` instead.
    """
    parsed = _parse_bucket_uri(argument)
    bucket_id = parsed.id
    prefix = parsed.path_in_repo

    if prefix == "" and not recursive:
        raise click.BadParameter(
            f"No file path specified. To remove files, provide a path"
            f" (e.g. '{bucket_id}/FILE') or use --recursive to remove all files."
            f" To delete the entire bucket, use `hf buckets delete {bucket_id}`."
        )

    if (include or exclude) and not recursive:
        raise click.BadParameter("--include and --exclude require --recursive.")

    api = get_hf_api(token=token)

    if recursive:
        status = out.status("Listing files from remote")

        all_files: list[BucketFile] = []
        for item in _iter_under_prefix(
            api.list_bucket_tree(
                bucket_id,
                prefix=prefix or None,
                recursive=True,
            ),
            prefix,
        ):
            if isinstance(item, BucketFile):
                all_files.append(item)
                status.update(f"Listing files from remote ({len(all_files)} files)")
        status.done(f"Listing files from remote ({len(all_files)} files)")

        if include or exclude:
            matcher = FilterMatcher(include_patterns=include, exclude_patterns=exclude)
            matched_files = [f for f in all_files if matcher.matches(f.path)]
        else:
            matched_files = all_files

        file_paths = [f.path for f in matched_files]
        total_size = sum(f.size for f in matched_files)
        size_str = format_size(total_size, human_readable=True)

        if not file_paths:
            out.text("No files to remove.")
            return

        count_label = f"{len(file_paths)} file(s) totaling {size_str}"

        if not yes and not dry_run:
            out.text("\n".join(f"  {path}" for path in file_paths))
            out.confirm(f"Remove {count_label} from '{bucket_id}'?", yes=False)

        if dry_run:
            out.text("\n".join(f"delete: {BUCKET_PREFIX}{bucket_id}/{path}" for path in file_paths))
            out.text(f"(dry run) {count_label} would be removed.")
            return

        api.batch_bucket_files(bucket_id, delete=file_paths)
        out.result(
            f"Removed {count_label} from '{bucket_id}'",
            bucket_id=bucket_id,
            files_deleted=len(file_paths),
            size=size_str,
        )

    else:
        file_path = prefix
        if not file_path:
            raise click.BadParameter("File path cannot be empty.")

        if dry_run:
            out.text(f"delete: {BUCKET_PREFIX}{bucket_id}/{file_path}")
            out.text("(dry run) 1 file would be removed.")
            return

        out.confirm(f"Remove '{file_path}' from '{bucket_id}'?", yes=yes)

        api.batch_bucket_files(bucket_id, delete=[file_path])
        out.result("File removed", path=file_path, bucket_id=bucket_id)


@buckets_cli.command(
    name="move",
    examples=[
        "hf buckets move user/old-bucket user/new-bucket",
        "hf buckets move user/my-bucket my-org/my-bucket",
        "hf buckets move hf://buckets/user/old-bucket hf://buckets/user/new-bucket",
    ],
)
def move(
    from_id: Annotated[
        str,
        Argument(
            help="Source bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    to_id: Annotated[
        str,
        Argument(
            help="Destination bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    token: TokenOpt = None,
) -> None:
    """Move (rename) a bucket to a new name or namespace."""
    # Parse from_id
    parsed_from = _parse_bucket_uri(from_id)
    if parsed_from.path_in_repo:
        raise click.BadParameter(
            f"Cannot specify a prefix for bucket move: {from_id}."
            f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
        )

    # Parse to_id
    parsed_to = _parse_bucket_uri(to_id)
    if parsed_to.path_in_repo:
        raise click.BadParameter(
            f"Cannot specify a prefix for bucket move: {to_id}."
            f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
        )

    api = get_hf_api(token=token)
    api.move_bucket(from_id=parsed_from.id, to_id=parsed_to.id)
    out.result("Bucket moved", from_id=parsed_from.id, to_id=parsed_to.id)


@buckets_cli.command(
    name="settings",
    examples=[
        "hf buckets settings user/my-bucket --private",
        "hf buckets settings user/my-bucket --public",
        "hf buckets settings hf://buckets/user/my-bucket --private",
    ],
)
def settings(
    bucket_id: Annotated[
        str,
        Argument(
            help="Bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    private: Annotated[
        bool,
        Option(
            "--private",
            help="Make the bucket private.",
        ),
    ] = False,
    public: Annotated[
        bool,
        Option(
            "--public",
            help="Make the bucket public.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Update bucket settings (visibility)."""
    if private == public:
        raise click.BadParameter("Specify exactly one of --private or --public.")

    if bucket_id.startswith(BUCKET_PREFIX):
        parsed = _parse_bucket_uri(bucket_id)
        if parsed.path_in_repo:
            raise click.BadParameter(
                f"Cannot specify a prefix for bucket settings: {bucket_id}."
                f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
            )
        bucket_id = parsed.id
    elif "/" not in bucket_id:
        raise click.BadParameter(
            f"Invalid bucket ID: {bucket_id}."
            f" Must be in format namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
        )

    api = get_hf_api(token=token)
    api.update_bucket_settings(bucket_id, private=private)
    out.result("Bucket settings updated", bucket_id=bucket_id, private=private)
    out.hint(f"Run `hf buckets info {bucket_id}` to view bucket details.")


# =============================================================================
# Sync command
# =============================================================================


@buckets_cli.command(
    name="sync",
    examples=[
        "hf buckets sync ./data hf://buckets/user/my-bucket",
        "hf buckets sync hf://buckets/user/my-bucket ./data",
        "hf buckets sync ./data hf://buckets/user/my-bucket --delete",
        'hf buckets sync hf://buckets/user/my-bucket ./data --include "*.safetensors" --exclude "*.tmp"',
        "hf buckets sync ./data hf://buckets/user/my-bucket --plan sync-plan.jsonl",
        "hf buckets sync --apply sync-plan.jsonl",
        "hf buckets sync ./data hf://buckets/user/my-bucket --dry-run",
        "hf buckets sync ./data hf://buckets/user/my-bucket --dry-run | jq .",
    ],
)
def sync(
    source: Annotated[
        str | None,
        Argument(
            help="Source path: local directory or hf://buckets/namespace/bucket_name(/prefix)",
        ),
    ] = None,
    dest: Annotated[
        str | None,
        Argument(
            help="Destination path: local directory or hf://buckets/namespace/bucket_name(/prefix)",
        ),
    ] = None,
    delete: Annotated[
        bool,
        Option(
            help="Delete destination files not present in source.",
        ),
    ] = False,
    ignore_times: Annotated[
        bool,
        Option(
            "--ignore-times",
            help="Skip files only based on size, ignoring modification times.",
        ),
    ] = False,
    ignore_sizes: Annotated[
        bool,
        Option(
            "--ignore-sizes",
            help="Skip files only based on modification times, ignoring sizes.",
        ),
    ] = False,
    plan: Annotated[
        str | None,
        Option(
            help="Save sync plan to JSONL file for review instead of executing.",
        ),
    ] = None,
    apply: Annotated[
        str | None,
        Option(
            help="Apply a previously saved plan file.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Option(
            "--dry-run",
            help="Print sync plan to stdout as JSONL without executing.",
        ),
    ] = False,
    include: Annotated[
        list[str] | None,
        Option(
            help="Include files matching pattern (can specify multiple).",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option(
            help="Exclude files matching pattern (can specify multiple).",
        ),
    ] = None,
    filter_from: Annotated[
        str | None,
        Option(
            help="Read include/exclude patterns from file.",
        ),
    ] = None,
    existing: Annotated[
        bool,
        Option(
            "--existing",
            help="Skip creating new files on receiver (only update existing files).",
        ),
    ] = False,
    ignore_existing: Annotated[
        bool,
        Option(
            "--ignore-existing",
            help="Skip updating files that exist on receiver (only create new files).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        Option(
            "--verbose",
            "-v",
            help="Show detailed logging with reasoning.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Sync files between local directory and a bucket."""
    api = get_hf_api(token=token)
    api.sync_bucket(
        source=source,
        dest=dest,
        delete=delete,
        ignore_times=ignore_times,
        ignore_sizes=ignore_sizes,
        existing=existing,
        ignore_existing=ignore_existing,
        include=include,
        exclude=exclude,
        filter_from=filter_from,
        plan=plan,
        apply=apply,
        dry_run=dry_run,
        verbose=verbose,
        quiet=out.is_quiet(),
    )
    if plan and not out.is_quiet():
        out.hint(f"Run `hf buckets sync --apply {plan}` to execute this plan.")


# =============================================================================
# Cp command
# =============================================================================


# `hf buckets cp` is an alias for the top-level `hf cp` command (see `cli/_cp.py`).
buckets_cli.command(
    name="cp",
    examples=[
        # Download (repo or bucket -> local / stdout)
        "hf buckets cp hf://buckets/username/my-bucket/config.json config.json",
        "hf buckets cp hf://buckets/username/my-bucket/data.csv data/",
        "hf buckets cp hf://buckets/username/my-bucket/config.json -",
        # Upload (local / stdin -> bucket)
        "hf buckets cp model.safetensors hf://buckets/username/my-bucket/model.safetensors",
        "hf buckets cp config.json hf://buckets/username/my-bucket/logs/",
        "hf buckets cp - hf://buckets/username/my-bucket/config.json",
        # Remote to remote (repo or bucket -> bucket)
        "hf buckets cp hf://buckets/username/my-bucket/data.csv hf://buckets/username/dest-bucket/",
        "hf buckets cp hf://buckets/username/source-bucket/logs/ hf://buckets/username/dest-bucket/logs/",
    ],
)(make_cp("buckets"))
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
"""Shared helpers for listing files in buckets and repos (tree view, flat view, formatting)."""

import json
from datetime import datetime
from typing import Sequence

import click

from huggingface_hub._buckets import BucketFile, BucketFolder
from huggingface_hub.hf_api import RepoFile, RepoFolder

from ._cli_utils import get_hf_api
from ._output import OutputFormat, _ascii_safe, _dataclass_to_dict, _escape_control_chars, out


BucketItem = BucketFile | BucketFolder
RepoItem = RepoFile | RepoFolder
ListingItem = BucketItem | RepoItem


def get_item_date(item: ListingItem) -> datetime | None:
    """Extract date from an item, supporting both repo items (last_commit.date) and bucket items (mtime/uploaded_at)."""
    match item:
        case BucketFile(mtime=mtime) if mtime is not None:
            return mtime
        case BucketFile(uploaded_at=uploaded_at) | BucketFolder(uploaded_at=uploaded_at) if uploaded_at is not None:
            return uploaded_at
        case RepoFile(last_commit=last_commit) | RepoFolder(last_commit=last_commit) if last_commit is not None:
            return last_commit.date
        case _:
            return None


def format_size(size: int | float, human_readable: bool = False) -> str:
    """Format a size in bytes."""
    if not human_readable:
        return str(size)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1000:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} PB"


def format_date(dt: datetime | None, human_readable: bool = False) -> str:
    """Format a datetime to a readable date string."""
    if dt is None:
        return ""
    if human_readable:
        return dt.strftime("%b %d %H:%M")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_tree(
    items: Sequence[BucketItem] | Sequence[RepoItem],
    human_readable: bool = False,
    quiet: bool = False,
) -> list[str]:
    """Build a tree representation of files and directories.

    Produces ASCII tree with size and date columns before the tree connector.
    When quiet=True, only the tree structure is shown (no size/date).
    """
    tree: dict = {}

    for item in items:
        parts = item.path.split("/")
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"__children__": {}}
            current = current[part]["__children__"]

        final_part = parts[-1]
        if isinstance(item, BucketFolder | RepoFolder):
            if final_part not in current:
                current[final_part] = {"__children__": {}}
        else:
            current[final_part] = {"__item__": item}

    prefix_width = 0
    max_size_width = 0
    max_date_width = 0
    if not quiet:
        for item in items:
            if isinstance(item, BucketFile | RepoFile):
                size_str = format_size(item.size, human_readable)
                max_size_width = max(max_size_width, len(size_str))
                date_str = format_date(get_item_date(item), human_readable)
                max_date_width = max(max_date_width, len(date_str))
        if max_size_width > 0:
            prefix_width = max_size_width + 2 + max_date_width

    lines: list[str] = []
    _render_tree(
        tree,
        lines,
        "",
        prefix_width=prefix_width,
        max_size_width=max_size_width,
        human_readable=human_readable,
    )
    return lines


def _render_tree(
    node: dict,
    lines: list[str],
    indent: str,
    prefix_width: int = 0,
    max_size_width: int = 0,
    human_readable: bool = False,
) -> None:
    """Recursively render a tree structure with size+date prefix."""
    sorted_items = sorted(node.items())
    for i, (name, value) in enumerate(sorted_items):
        is_last = i == len(sorted_items) - 1
        connector = _ascii_safe("└── ", "`-- ") if is_last else _ascii_safe("├── ", "|-- ")

        is_dir = "__children__" in value
        children = value.get("__children__", {})

        if prefix_width > 0:
            if is_dir:
                prefix = " " * prefix_width
            else:
                item = value.get("__item__")
                if item is not None:
                    size_str = format_size(item.size, human_readable)
                    date_str = format_date(get_item_date(item), human_readable)
                    prefix = f"{size_str:>{max_size_width}}  {date_str}"
                else:
                    prefix = " " * prefix_width
            lines.append(f"{prefix}  {indent}{connector}{_escape_control_chars(name)}{'/' if is_dir else ''}")
        else:
            lines.append(f"{indent}{connector}{_escape_control_chars(name)}{'/' if is_dir else ''}")

        if children:
            child_indent = indent + ("    " if is_last else _ascii_safe("│   ", "|   "))
            _render_tree(
                children,
                lines,
                child_indent,
                prefix_width=prefix_width,
                max_size_width=max_size_width,
                human_readable=human_readable,
            )


def list_repo_files_cmd(
    repo_id: str,
    repo_type: str,
    human_readable: bool,
    as_tree: bool,
    recursive: bool,
    revision: str | None,
    token: str | None,
) -> None:
    """List files in a repo on the Hub. Used by models/datasets/spaces ls commands."""
    if as_tree and out.mode == OutputFormat.json:
        raise click.BadParameter("Cannot use --tree with --format json.")

    api = get_hf_api(token=token)
    items = list(api.list_repo_tree(repo_id, recursive=recursive, revision=revision, repo_type=repo_type, expand=True))
    print_file_listing(items, human_readable=human_readable, as_tree=as_tree, recursive=recursive)


def print_file_listing(
    items: Sequence[BucketItem] | Sequence[RepoItem],
    *,
    human_readable: bool = False,
    as_tree: bool = False,
    recursive: bool = False,
) -> None:
    """Print a file listing in the appropriate format based on the current output mode.

    Supports tree, json, quiet, and flat human-readable views. Works with both
    BucketFile/BucketFolder and RepoFile/RepoFolder items.
    """
    if not items:
        out.text("(empty)")
        return

    has_directories = any(isinstance(item, BucketFolder | RepoFolder) for item in items)

    if as_tree:
        quiet = out.mode == OutputFormat.quiet
        for line in build_tree(items, human_readable=human_readable, quiet=quiet):
            print(line)
    elif out.mode == OutputFormat.json:
        print(json.dumps([_dataclass_to_dict(item) for item in items], indent=2))
    elif out.mode == OutputFormat.quiet:
        for item in items:
            if isinstance(item, BucketFolder | RepoFolder):
                print(f"{_escape_control_chars(item.path)}/")
            else:
                print(_escape_control_chars(item.path))
    else:
        for item in items:
            if isinstance(item, BucketFolder | RepoFolder):
                date_str = format_date(get_item_date(item), human_readable)
                print(f"{'':>12}  {date_str:>19}  {_escape_control_chars(item.path)}/")
            else:
                size_str = format_size(item.size, human_readable)
                date_str = format_date(get_item_date(item), human_readable)
                print(f"{size_str:>12}  {date_str:>19}  {_escape_control_chars(item.path)}")

    if not recursive and has_directories:
        out.hint("Use -R to list files recursively.")
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
"""Minimal declaration layer over Click 8.x for the ``hf`` CLI.

This module vendors *only* the slice of Typer the CLI actually uses: turning
annotated function signatures (``Annotated[T, Option(...)]`` / ``Argument(...)``)
into Click parameters, and rendering argument/option help the way Typer did.
It exists so the CLI depends on Click's stable public API instead of Typer's
now-internal vendored Click (``typer._click``), which Typer reserves the right
to refactor at any release.

Deliberately *not* implemented (add a case here only when a command needs it):
rich rendering, shell-completion install commands, prompts/confirmation options,
env-var options, ``File``/``UUID`` params, ``count`` options, ``default_factory``.
Click's own machinery already handles exceptions, aborts, exits, and native
``_HF_COMPLETE`` shell completion, so none of Typer's ``main()`` overrides are
reproduced here.

The interesting logic mirrors four Typer functions, kept faithful so ``--help``
output stays byte-identical:
- :func:`_get_click_type`          <- ``typer.main.get_click_type``
- :func:`_build_click_param`       <- ``typer.main.get_click_param``
- :func:`_make_handler`            <- ``typer.main.get_callback``
- :class:`HfArgument` / :class:`HfOption` help records <- ``typer.core.Typer*``
"""

import enum
import functools
import inspect
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import click


def get_command_name(name: str) -> str:
    """Normalize a Python identifier to a CLI name (``force_download`` -> ``force-download``)."""
    return name.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# 1. Declaration markers — replace ``typer.Option`` / ``typer.Argument``.
#    Attached in ``Annotated[T, Option(...)]``; the ``= value`` in the function
#    signature carries the default (never the marker).
# ---------------------------------------------------------------------------


class ParameterInfo:
    """Base marker holding the (tiny) set of Click knobs the CLI actually sets.

    ``callback`` is a Typer-style value parser ``fn(value) -> parsed`` — it does NOT
    receive Click's ``(ctx, param, value)``. It consumes the enum/list convertors, so
    ``value`` is already converted when it runs.
    """

    def __init__(
        self,
        *param_decls: str,
        help: str | None = None,
        show_default: bool | str = True,
        hidden: bool = False,
        is_eager: bool = False,
        callback: Callable[..., Any] | None = None,
        min: int | None = None,
        click_type: click.ParamType | None = None,
    ) -> None:
        self.param_decls = list(param_decls)
        self.help = help
        self.show_default = show_default
        self.hidden = hidden
        self.is_eager = is_eager
        self.callback = callback
        self.min = min
        self.click_type = click_type


class Option(ParameterInfo):
    """Declare a CLI option, e.g. ``Annotated[str | None, Option("--revision")]``."""


class Argument(ParameterInfo):
    """Declare a positional CLI argument, e.g. ``Annotated[str, Argument()]``."""


# ---------------------------------------------------------------------------
# 2. Annotation -> Click type + value convertors.
# ---------------------------------------------------------------------------


def _get_click_type(annotation: Any, info: ParameterInfo) -> click.ParamType:
    """Map a (already unwrapped) annotation to a Click ``ParamType``."""
    if info.click_type is not None:
        return info.click_type
    if annotation is str:
        return click.STRING
    if annotation is bool:
        return click.BOOL
    if annotation is int:
        return click.IntRange(min=info.min) if info.min is not None else click.INT
    if annotation is float:
        return click.FloatRange(min=info.min) if info.min is not None else click.FLOAT
    if annotation is Path:
        return click.Path(path_type=Path)
    if annotation is datetime:
        return click.DateTime()
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [item.value for item in annotation]
        if not all(isinstance(value, str) for value in values):
            # A non-str member default would fail Choice validation at parse time with a
            # cryptic error; fail at build time with a clear one instead.
            raise TypeError(f"Unsupported enum {annotation.__name__!r}: only str-valued enums are supported.")
        return click.Choice(values)
    if get_origin(annotation) is Literal:
        return click.Choice([str(value) for value in get_args(annotation)])
    raise TypeError(f"Unsupported CLI annotation {annotation!r}. Add a case in `_framework._get_click_type()`.")


def _enum_convertor(enum_type: type[enum.Enum]) -> Callable[[Any], Any]:
    """Map a parsed choice string back to its ``Enum`` member (mirrors Typer)."""
    value_map = {str(member.value): member for member in enum_type}

    def convertor(value: Any) -> Any:
        if value is not None:
            return value_map.get(str(value))
        return None

    return convertor


def _list_convertor(
    inner: Callable[[Any], Any] | None, default_value: Any | None
