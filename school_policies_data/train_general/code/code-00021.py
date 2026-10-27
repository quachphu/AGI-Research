    file is uploaded (i.e. the part that was available when the filesystem was first scanned).

    In practice, only used internally by the CommitScheduler to regularly push a folder to the Hub with minimal
    disturbance for the user. The object is passed to `CommitOperationAdd`.

    Only supports `read`, `tell` and `seek` methods.

    Args:
        file_path (`str` or `Path`):
            Path to the file to read.
        size_limit (`int`):
            The maximum number of bytes to read from the file. If the file is larger than this, only the first part
            will be read (and uploaded).
    """

    def __init__(self, file_path: str | Path, size_limit: int) -> None:
        self._file_path = Path(file_path)
        self._file = self._file_path.open("rb")
        self._size_limit = min(size_limit, os.fstat(self._file.fileno()).st_size)

    def __del__(self) -> None:
        self._file.close()
        return super().__del__()

    def __repr__(self) -> str:
        return f"<PartialFileIO file_path={self._file_path} size_limit={self._size_limit}>"

    def __len__(self) -> int:
        return self._size_limit

    def __getattribute__(self, name: str):
        if name.startswith("_") or name in ("read", "tell", "seek", "fileno"):  # only 4 public methods supported
            return super().__getattribute__(name)
        raise NotImplementedError(f"PartialFileIO does not support '{name}'.")

    def fileno(self):
        raise AttributeError("PartialFileIO does not have a fileno.")

    def tell(self) -> int:
        """Return the current file position."""
        return self._file.tell()

    def seek(self, __offset: int, __whence: int = SEEK_SET) -> int:
        """Change the stream position to the given offset.

        Behavior is the same as a regular file, except that the position is capped to the size limit.
        """
        if __whence == SEEK_END:
            # SEEK_END => set from the truncated end
            __offset = len(self) + __offset
            __whence = SEEK_SET

        pos = self._file.seek(__offset, __whence)
        if pos > self._size_limit:
            return self._file.seek(self._size_limit)
        return pos

    def read(self, __size: int | None = -1) -> bytes:
        """Read at most `__size` bytes from the file.

        Behavior is the same as a regular file, except that it is capped to the size limit.
        """
        current = self._file.tell()
        if __size is None or __size < 0:
            # Read until file limit
            truncated_size = self._size_limit - current
        else:
            # Read until file limit or __size
            truncated_size = min(__size, self._size_limit - current)
        return self._file.read(truncated_size)
<|endoftext|>import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from huggingface_hub.errors import InferenceEndpointError, InferenceEndpointTimeoutError

from .utils import get_session, logging, parse_datetime
from .utils._headers import get_token_to_send


if TYPE_CHECKING:
    from .hf_api import HfApi
    from .inference._client import InferenceClient
    from .inference._generated._async_client import AsyncInferenceClient

logger = logging.get_logger(__name__)


def _build_endpoint_image_payload(
    custom_image: dict,
    *,
    container_registry_username: str | None = None,
    container_registry_password: str | None = None,
) -> dict:
    """Build the `model.image` payload of an Inference Endpoint from a user-provided image dict.

    `model.image` is a union keyed by variant (`{"vLLM": {...}}`, `{"custom": {...}}`, ...). Only a flat
    container dict has a top-level `url` (required server-side, and no variant is named `url`), so dicts with
    one are wrapped in `{"custom": ...}`. Everything else is forwarded as-is, so variants added to the API
    later work without a release.
    """
    image = {"custom": custom_image} if "url" in custom_image else custom_image

    if container_registry_password is not None and container_registry_username is None:
        raise ValueError("`container_registry_password` requires `container_registry_username`.")
    if container_registry_username is None:
        return image

    custom_container = image.get("custom")
    if not isinstance(custom_container, dict):
        raise ValueError("Container registry credentials can only be set for a custom container image.")

    credentials = {"username": container_registry_username}
    if container_registry_password is not None:
        credentials["password"] = container_registry_password
    return {**image, "custom": {**custom_container, "credentials": credentials}}


# Image variants that declare `tensorParallelSize` / `dataParallelSize`. The API ignores a field a variant doesn't
# declare instead of rejecting it, so these only drive a warning: a wrong engine gets a visible no-op rather than a
# silent one, and an engine the API adds later still works without a `huggingface_hub` release.
_TENSOR_PARALLEL_IMAGE_KEYS = ("sGLang", "vLLM")
_DATA_PARALLEL_IMAGE_KEYS = ("vLLM",)


def _set_parallelism_in_image(
    image: dict,
    *,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
) -> dict:
    """Write the parallelism sizes into a `model.image` payload.

    They are engine settings, so they live inside the engine config (`{"vLLM": {"url": ..., "tensorParallelSize": 8}}`)
    rather than at the top level. Returns a new image dict, the input is left untouched.
    """
    image_key = next(iter(image), None)
    if image_key is None:
        raise ValueError("Cannot set the parallelism sizes: the image payload is empty.")

    for value, name, supported in (
        (tensor_parallel_size, "tensor_parallel_size", _TENSOR_PARALLEL_IMAGE_KEYS),
        (data_parallel_size, "data_parallel_size", _DATA_PARALLEL_IMAGE_KEYS),
    ):
        if value is not None and image_key not in supported:
            warnings.warn(
                f"`{name}` is not a known setting of the '{image_key}' image: the API will silently drop it."
                f" Engines that support it: {', '.join(supported)}.",
                UserWarning,
            )

    image = {image_key: {**image[image_key]}}
    if tensor_parallel_size is not None:
        image[image_key]["tensorParallelSize"] = tensor_parallel_size
    if data_parallel_size is not None:
        image[image_key]["dataParallelSize"] = data_parallel_size
    return image


class InferenceEndpointStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    UPDATING = "updating"
    UPDATE_FAILED = "updateFailed"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    SCALED_TO_ZERO = "scaledToZero"


class InferenceEndpointType(str, Enum):
    PUBlIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVATE = "private"


class InferenceEndpointScalingMetric(str, Enum):
    PENDING_REQUESTS = "pendingRequests"
    HARDWARE_USAGE = "hardwareUsage"


@dataclass
class InferenceEndpoint:
    """
    Contains information about a deployed Inference Endpoint.

    Args:
        name (`str`):
            The unique name of the Inference Endpoint.
        namespace (`str`):
            The namespace where the Inference Endpoint is located.
        repository (`str`):
            The name of the model repository deployed on this Inference Endpoint.
        status ([`InferenceEndpointStatus`]):
            The current status of the Inference Endpoint.
        url (`str`, *optional*):
            The URL of the Inference Endpoint, if available. Only a deployed Inference Endpoint will have a URL.
        framework (`str`):
            The machine learning framework used for the model.
        revision (`str`):
            The specific model revision deployed on the Inference Endpoint.
        task (`str`):
            The task associated with the deployed model.
        created_at (`datetime.datetime`):
            The timestamp when the Inference Endpoint was created.
        updated_at (`datetime.datetime`):
            The timestamp of the last update of the Inference Endpoint.
        type ([`InferenceEndpointType`]):
            The type of the Inference Endpoint (public, authenticated, private).
        raw (`dict`):
            The raw dictionary data returned from the API.
        token (`str` or `bool`, *optional*):
            Authentication token for the Inference Endpoint, if set when requesting the API. Will default to the
            locally saved token if not provided. Pass `token=False` if you don't want to send your token to the server.

    Example:
        ```python
        >>> from huggingface_hub import get_inference_endpoint
        >>> endpoint = get_inference_endpoint("my-text-to-image")
        >>> endpoint
        InferenceEndpoint(name='my-text-to-image', ...)

        # Get status
        >>> endpoint.status
        'running'
        >>> endpoint.url
        'https://68a0f3b5c6ee5c9dcc0e0a01.endpoints.huggingface.cloud'

        # Run inference
        >>> endpoint.client.text_to_image(...)

        # Pause endpoint to save $$$
        >>> endpoint.pause()

        # ...
        # Resume and wait for deployment
        >>> endpoint.resume()
        >>> endpoint.wait()
        >>> endpoint.client.text_to_image(...)
        ```
    """

    # Field in __repr__
    name: str = field(init=False)
    namespace: str
    repository: str = field(init=False)
    status: InferenceEndpointStatus = field(init=False)
    health_route: str = field(init=False)
    url: str | None = field(init=False)

    # Other fields
    framework: str = field(repr=False, init=False)
    revision: str = field(repr=False, init=False)
    task: str = field(repr=False, init=False)
    created_at: datetime = field(repr=False, init=False)
    updated_at: datetime = field(repr=False, init=False)
    type: InferenceEndpointType = field(repr=False, init=False)

    # Raw dict from the API
    raw: dict = field(repr=False)

    # Internal fields
    _token: str | bool | None = field(repr=False, compare=False)
    _api: "HfApi" = field(repr=False, compare=False)

    @classmethod
    def from_raw(
        cls, raw: dict, namespace: str, token: str | bool | None = None, api: Optional["HfApi"] = None
    ) -> "InferenceEndpoint":
        """Initialize object from raw dictionary."""
        if api is None:
            from .hf_api import HfApi

            api = HfApi()
        if token is None:
            token = api.token
        if token is True:
            token = get_token_to_send(token)

        # All other fields are populated in __post_init__
        return cls(raw=raw, namespace=namespace, _token=token, _api=api)

    def __post_init__(self) -> None:
        """Populate fields from raw dictionary."""
        self._populate_from_raw()

    @property
    def client(self) -> "InferenceClient":
        """Returns a client to make predictions on this Inference Endpoint.

        Returns:
            [`InferenceClient`]: an inference client pointing to the deployed endpoint.

        Raises:
            [`InferenceEndpointError`]: If the Inference Endpoint is not yet deployed.
        """
        if self.url is None:
            raise InferenceEndpointError(
                "Cannot create a client for this Inference Endpoint as it is not yet deployed. "
                "Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again."
            )
        from .inference._client import InferenceClient

        return InferenceClient(
            model=self.url,
            token=self._token,  # type: ignore # boolean token shouldn't be possible. In practice it's ok.
        )

    @property
    def async_client(self) -> "AsyncInferenceClient":
        """Returns a client to make predictions on this Inference Endpoint.

        Returns:
            [`AsyncInferenceClient`]: an asyncio-compatible inference client pointing to the deployed endpoint.

        Raises:
            [`InferenceEndpointError`]: If the Inference Endpoint is not yet deployed.
        """
        if self.url is None:
            raise InferenceEndpointError(
                "Cannot create a client for this Inference Endpoint as it is not yet deployed. "
                "Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again."
            )
        from .inference._generated._async_client import AsyncInferenceClient

        return AsyncInferenceClient(
            model=self.url,
            token=self._token,  # type: ignore # boolean token shouldn't be possible. In practice it's ok.
        )

    def wait(self, timeout: int | None = None, refresh_every: int = 5) -> "InferenceEndpoint":
        """Wait for the Inference Endpoint to be deployed.

        Information from the server will be fetched every 1s. If the Inference Endpoint is not deployed after `timeout`
        seconds, a [`InferenceEndpointTimeoutError`] will be raised. The [`InferenceEndpoint`] will be mutated in place with the latest
        data.

        Args:
            timeout (`int`, *optional*):
                The maximum time to wait for the Inference Endpoint to be deployed, in seconds. If `None`, will wait
                indefinitely.
            refresh_every (`int`, *optional*):
                The time to wait between each fetch of the Inference Endpoint status, in seconds. Defaults to 5s.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.

        Raises:
            [`InferenceEndpointError`]
                If the Inference Endpoint ended up in a failed state.
            [`InferenceEndpointTimeoutError`]
                If the Inference Endpoint is not deployed after `timeout` seconds.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("`timeout` cannot be negative.")
        if refresh_every <= 0:
            raise ValueError("`refresh_every` must be positive.")

        start = time.time()
        while True:
            if self.status == InferenceEndpointStatus.FAILED:
                raise InferenceEndpointError(
                    f"Inference Endpoint {self.name} failed to deploy. Please check the logs for more information."
                )
            if self.status == InferenceEndpointStatus.UPDATE_FAILED:
                raise InferenceEndpointError(
                    f"Inference Endpoint {self.name} failed to update. Please check the logs for more information."
                )
            if self.status == InferenceEndpointStatus.RUNNING and self.url is not None:
                # Verify the endpoint is actually reachable
                _health_url = f"{self.url.rstrip('/')}/{self.health_route.lstrip('/')}"
                response = get_session().get(_health_url, headers=self._api._build_hf_headers(token=self._token))
                if response.status_code == 200:
                    logger.info("Inference Endpoint is ready to be used.")
                    return self

            if timeout is not None:
                if time.time() - start > timeout:
                    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
            logger.info(f"Inference Endpoint is not deployed yet ({self.status}). Waiting {refresh_every}s...")
            time.sleep(refresh_every)
            self.fetch()

    def fetch(self) -> "InferenceEndpoint":
        """Fetch latest information about the Inference Endpoint.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
        obj = self._api.get_inference_endpoint(name=self.name, namespace=self.namespace, token=self._token)  # type: ignore [arg-type]
        self.raw = obj.raw
        self._populate_from_raw()
        return self

    def update(
        self,
        *,
        # Compute update
        accelerator: str | None = None,
        instance_size: str | None = None,
        instance_type: str | None = None,
        min_replica: int | None = None,
        max_replica: int | None = None,
        scale_to_zero_timeout: int | None = None,
        # Model update
        repository: str | None = None,
        framework: str | None = None,
        revision: str | None = None,
        task: str | None = None,
        custom_image: dict | None = None,
        container_command: list[str] | None = None,
        container_args: list[str] | None = None,
        tensor_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
        secrets: dict[str, str] | None = None,
    ) -> "InferenceEndpoint":
        """Update the Inference Endpoint.

        This method allows the update of either the compute configuration, the deployed model, or both. All arguments are
        optional but at least one must be provided.

        This is an alias for [`HfApi.update_inference_endpoint`]. The current object is mutated in place with the
        latest data from the server.

        Args:
            accelerator (`str`, *optional*):
                The hardware accelerator to be used for inference (e.g. `"cpu"`).
            instance_size (`str`, *optional*):
                The size or type of the instance to be used for hosting the model (e.g. `"x4"`).
            instance_type (`str`, *optional*):
                The cloud instance type where the Inference Endpoint will be deployed (e.g. `"intel-icl"`).
            min_replica (`int`, *optional*):
                The minimum number of replicas (instances) to keep running for the Inference Endpoint.
            max_replica (`int`, *optional*):
                The maximum number of replicas (instances) to scale to for the Inference Endpoint.
            scale_to_zero_timeout (`int`, *optional*):
                The duration in minutes before an inactive endpoint is scaled to zero.

            repository (`str`, *optional*):
                The name of the model repository associated with the Inference Endpoint (e.g. `"gpt2"`).
            framework (`str`, *optional*):
                The machine learning framework used for the model (e.g. `"custom"`).
            revision (`str`, *optional*):
                The specific model revision to deploy on the Inference Endpoint (e.g. `"6c0e6080953db56375760c0471a8c5f2929baf11"`).
            task (`str`, *optional*):
                The task on which to deploy the model (e.g. `"text-classification"`).
            custom_image (`dict`, *optional*):
                The container image to run. Either a dict keyed by image variant (e.g.
                `{"vLLM": {"url": "vllm/vllm-openai:v0.23.0", "port": 8000}}`, also `sGLang`, `tgi`, `tei`,
                `llamacpp`, `hfServe`, ...), which is forwarded as-is, or a flat dict describing a custom
                container (e.g. `{"url": ..., "port": ...}`), which is sent as `{"custom": ...}`.
            container_command (`list[str]`, *optional*):
                Override the container entrypoint command (maps to `model.command` in the API payload). Works with
                both managed engine images (e.g. vLLM, SGLang) and custom images.
            container_args (`list[str]`, *optional*):
                Arguments appended to the container entrypoint (maps to `model.args` in the API payload). Works with
                both managed engine images (e.g. vLLM, SGLang) and custom images.
            tensor_parallel_size (`int`, *optional*):
                Number of accelerators to shard a single model copy across (vLLM and SGLang images). When
                `custom_image` is not given, the image currently configured on the endpoint is fetched and updated
                in place, as the API requires `model.image` as a whole.
            data_parallel_size (`int`, *optional*):
                Number of model copies to run, one per accelerator (vLLM images).
            secrets (`dict[str, str]`, *optional*):
                Secret values to inject in the container environment.
        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
        # Make API call
        obj = self._api.update_inference_endpoint(
            name=self.name,
            namespace=self.namespace,
            accelerator=accelerator,
            instance_size=instance_size,
            instance_type=instance_type,
            min_replica=min_replica,
            max_replica=max_replica,
            scale_to_zero_timeout=scale_to_zero_timeout,
            repository=repository,
            framework=framework,
            revision=revision,
            task=task,
            custom_image=custom_image,
            container_command=container_command,
            container_args=container_args,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            secrets=secrets,
            token=self._token,  # type: ignore [arg-type]
        )

        # Mutate current object
        self.raw = obj.raw
        self._populate_from_raw()
        return self

    def pause(self) -> "InferenceEndpoint":
        """Pause the Inference Endpoint.

        A paused Inference Endpoint will not be charged. It can be resumed at any time using [`InferenceEndpoint.resume`].
        This is different from scaling the Inference Endpoint to zero with [`InferenceEndpoint.scale_to_zero`], which
        would be automatically restarted when a request is made to it.

        This is an alias for [`HfApi.pause_inference_endpoint`]. The current object is mutated in place with the
        latest data from the server.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
        obj = self._api.pause_inference_endpoint(name=self.name, namespace=self.namespace, token=self._token)  # type: ignore [arg-type]
        self.raw = obj.raw
        self._populate_from_raw()
        return self

    def resume(self, running_ok: bool = True) -> "InferenceEndpoint":
        """Resume the Inference Endpoint.

        This is an alias for [`HfApi.resume_inference_endpoint`]. The current object is mutated in place with the
        latest data from the server.

        Args:
            running_ok (`bool`, *optional*):
                If `True`, the method will not raise an error if the Inference Endpoint is already running. Defaults to
                `True`.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
        obj = self._api.resume_inference_endpoint(
            name=self.name, namespace=self.namespace, running_ok=running_ok, token=self._token
        )  # type: ignore [arg-type]
        self.raw = obj.raw
        self._populate_from_raw()
        return self

    def scale_to_zero(self) -> "InferenceEndpoint":
        """Scale Inference Endpoint to zero.

        An Inference Endpoint scaled to zero will not be charged. It will be resumed on the next request to it, with a
        cold start delay. This is different from pausing the Inference Endpoint with [`InferenceEndpoint.pause`], which
        would require a manual resume with [`InferenceEndpoint.resume`].

        This is an alias for [`HfApi.scale_to_zero_inference_endpoint`]. The current object is mutated in place with the
        latest data from the server.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
        obj = self._api.scale_to_zero_inference_endpoint(name=self.name, namespace=self.namespace, token=self._token)  # type: ignore [arg-type]
        self.raw = obj.raw
        self._populate_from_raw()
        return self

    def delete(self) -> None:
        """Delete the Inference Endpoint.

        This operation is not reversible. If you don't want to be charged for an Inference Endpoint, it is preferable
        to pause it with [`InferenceEndpoint.pause`] or scale it to zero with [`InferenceEndpoint.scale_to_zero`].

        This is an alias for [`HfApi.delete_inference_endpoint`].
        """
        self._api.delete_inference_endpoint(name=self.name, namespace=self.namespace, token=self._token)  # type: ignore [arg-type]

    def _populate_from_raw(self) -> None:
        """Populate fields from raw dictionary.

        Called in __post_init__ + each time the Inference Endpoint is updated.
        """
        # Repr fields
        self.name = self.raw["name"]
        self.repository = self.raw["model"]["repository"]
        self.status = self.raw["status"]["state"]
        self.url = self.raw["status"].get("url")
        self.health_route = self.raw["healthRoute"]

        # Other fields
        self.framework = self.raw["model"]["framework"]
        self.revision = self.raw["model"]["revision"]
        self.task = self.raw["model"]["task"]
        self.created_at = parse_datetime(self.raw["status"]["createdAt"])
        self.updated_at = parse_datetime(self.raw["status"]["updatedAt"])
        self.type = self.raw["type"]


@dataclass
class InferenceEndpointHardware:
    """
    Contains information about a hardware configuration available for Inference Endpoints.

    The `vendor`, `region`, `accelerator`, `instance_type` and `instance_size` fields are exactly the values to pass to
    [`create_inference_endpoint`] (or `hf endpoints deploy`) to deploy on this hardware.

    Args:
        id (`str`):
            Unique hardware identifier, e.g. `"aws-us-east-1-nvidia-l4-x1"`.
        vendor (`str`):
            The cloud provider hosting the hardware, e.g. `"aws"`.
        region (`str`):
            The cloud region the hardware is available in, e.g. `"us-east-1"`.
        accelerator (`str`):
            The type of hardware accelerator, e.g. `"cpu"`, `"gpu"` or `"neuron"`.
        instance_type (`str`):
            The cloud instance type, e.g. `"nvidia-l4"`.
        instance_size (`str`):
            The instance size multiplier, e.g. `"x1"`.
        architecture (`str`):
            Human-readable hardware description, e.g. `"Nvidia L4"`.
        num_accelerators (`int`):
            Number of accelerator units per replica.
        num_cpus (`int`, *optional*):
            Number of vCPUs per replica.
        memory_gb (`float`):
            RAM per replica, in GB.
        gpu_memory_gb (`int`, *optional*):
            Total GPU memory per replica, in GB (i.e. summed over `num_accelerators`). `None` for non-GPU hardware.
        price_per_hour (`float`):
            Cost per replica per hour, in USD.
        status (`str`):
            Availability of the hardware: `"available"`, `"low_availability"`, `"not_available"`, `"reserved"` or
            `"deprecated"`.
        max_accelerators (`int`):
            Maximum number of accelerators of this type the namespace is allowed to run.
        used_accelerators (`int`):
            Number of accelerators of this type currently used by the namespace.

    Example:
        ```python
        >>> from huggingface_hub import list_inference_endpoints_hardware
        >>> hardware = list_inference_endpoints_hardware()
        >>> hardware[0]
        InferenceEndpointHardware(id='aws-us-east-1-nvidia-l4-x1', vendor='aws', region='us-east-1', ...)
        ```
    """

    id: str
    vendor: str
    region: str
    accelerator: str
    instance_type: str
    instance_size: str
    architecture: str
    num_accelerators: int
    num_cpus: int | None
    memory_gb: float
    gpu_memory_gb: int | None
    price_per_hour: float
    status: str
    max_accelerators: int
    used_accelerators: int

    @classmethod
    def from_raw(cls, raw: dict, *, vendor: str, region: str) -> "InferenceEndpointHardware":
        """Initialize object from a raw compute dictionary, nested under a vendor and a region in the API response."""
        quota = raw["quota"]
        return cls(
            id=raw["id"],
            vendor=vendor,
            region=region,
            accelerator=raw["accelerator"],
            instance_type=raw["instanceType"],
            instance_size=raw["instanceSize"],
            architecture=raw["architecture"],
            num_accelerators=raw["numAccelerators"],
            num_cpus=raw.get("numCpus"),
            memory_gb=raw["memoryGb"],
            gpu_memory_gb=raw.get("gpuMemoryGb"),
            price_per_hour=raw["pricePerHour"],
            status=raw["status"],
            max_accelerators=quota["maxAccelerators"],
            used_accelerators=quota["usedAccelerators"],
        )


@dataclass
class InferenceCatalogRecipe:
    """
    Contains information about a pre-tested way of deploying a model from the Inference Catalog.

    A recipe is a hardware and engine combination that the Inference Endpoints team has tested for a given model. A
    model usually has one recipe, but it can have several (e.g. a GPU one and a Neuron one, or one per GGUF quant).

    Args:
        id (`str`):
            The public recipe id, to pass to [`create_inference_endpoint_from_catalog`] as `recipe_id`.
        accelerator (`str`):
            The type of hardware accelerator the recipe runs on, e.g. `"cpu"`, `"gpu"` or `"neuron"`.
        engine (`str`):
            The inference engine the recipe runs, e.g. `"vllm"`, `"llamacpp"`, `"tei"` or `"sglang"`.
        gguf_file (`str`, *optional*):
            The GGUF file deployed by the recipe. Only set for `"llamacpp"` recipes.
        revision (`str`, *optional*):
            The model revision the recipe is pinned to, if any.
    """

    id: str
    accelerator: str
    engine: str
    gguf_file: str | None
    revision: str | None

    @classmethod
    def from_raw(cls, raw: dict) -> "InferenceCatalogRecipe":
        """Initialize object from a raw recipe dictionary, nested under a model in the API response."""
        return cls(
            id=raw["publicId"],
            accelerator=raw["accelerator"],
            engine=raw["engineType"],
            gguf_file=raw.get("ggufFile"),
            revision=raw.get("revision"),
        )


@dataclass
class InferenceCatalogModel:
    """
    Contains information about a model available in the Inference Catalog.

    Args:
        repo_id (`str`):
            The id of the model repository on the Hub, e.g. `"meta-llama/Llama-3.1-8B-Instruct"`.
        name (`str`):
            The model name, i.e. the part of `repo_id` after the author, e.g. `"Llama-3.1-8B-Instruct"`.
        author (`str`):
            The author of the model repository, e.g. `"meta-llama"`.
        license (`str`):
            The license of the model, e.g. `"Apache 2.0"`.
        task (`str`):
            The task the model is deployed on, e.g. `"text-generation"`.
        created_at (`datetime`):
            The date the model was added to the catalog.
        recipes (`list[InferenceCatalogRecipe]`):
            The tested ways of deploying this model.

    Example:
        ```python
        >>> from huggingface_hub import list_inference_catalog
        >>> catalog = list_inference_catalog(engine="vllm")
        >>> catalog[0]
        InferenceCatalogModel(repo_id='Qwen/Qwen2.5-Coder-7B-Instruct', name='Qwen2.5-Coder-7B-Instruct', ...)
        ```
    """

    repo_id: str
    name: str
    author: str
    license: str
    task: str
    created_at: datetime
    recipes: list[InferenceCatalogRecipe]

    @classmethod
    def from_raw(cls, raw: dict) -> "InferenceCatalogModel":
        """Initialize object from a raw catalog item dictionary."""
        return cls(
            repo_id=raw["repoId"],
            name=raw["modelName"],
            author=raw["authorName"],
            license=raw["license"],
            task=raw["task"],
            created_at=parse_datetime(raw["createdAt"]),
            recipes=[InferenceCatalogRecipe.from_raw(recipe) for recipe in raw["recipes"]],
        )
<|endoftext|>import collections.abc
import inspect
import types
from collections.abc import Callable
from dataclasses import MISSING, Field, field, fields, make_dataclass
from functools import lru_cache, wraps
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)


try:
    # Python 3.11+
    from typing import NotRequired, Required  # type: ignore
except ImportError:
    try:
        # In case typing_extensions is installed
        from typing_extensions import NotRequired, Required  # type: ignore
    except ImportError:
        # Fallback: create dummy types that will never match
        Required = type("Required", (), {})  # type: ignore
        NotRequired = type("NotRequired", (), {})  # type: ignore

from .errors import (
    StrictDataclassClassValidationError,
    StrictDataclassDefinitionError,
    StrictDataclassFieldValidationError,
)


Validator_T = Callable[[Any], None]
T = TypeVar("T")
TypedDictType = TypeVar("TypedDictType", bound=dict[str, Any])

_TYPED_DICT_DEFAULT_VALUE = object()  # used as default value in TypedDict fields (to distinguish from None)


# The overload decorator helps type checkers understand the different return types
@overload
def strict(cls: Type[T]) -> Type[T]: ...


@overload
def strict(*, accept_kwargs: bool = False) -> Callable[[Type[T]], Type[T]]: ...


def strict(cls: Type[T] | None = None, *, accept_kwargs: bool = False) -> Type[T] | Callable[[Type[T]], Type[T]]:
    """
    Decorator to add strict validation to a dataclass.

    This decorator must be used on top of `@dataclass` to ensure IDEs and static typing tools
    recognize the class as a dataclass.

    Can be used with or without arguments:
    - `@strict`
    - `@strict(accept_kwargs=True)`

    Args:
        cls:
            The class to convert to a strict dataclass.
        accept_kwargs (`bool`, *optional*):
            If True, allows arbitrary keyword arguments in `__init__`. Defaults to False.

    Returns:
        The enhanced dataclass with strict validation on field assignment.

    Example:
    ```py
    >>> from dataclasses import dataclass
    >>> from huggingface_hub.dataclasses import as_validated_field, strict, validated_field

    >>> @as_validated_field
    >>> def positive_int(value: int):
    ...     if not value >= 0:
    ...         raise ValueError(f"Value must be positive, got {value}")

    >>> @strict(accept_kwargs=True)
    ... @dataclass
    ... class User:
    ...     name: str
    ...     age: int = positive_int(default=10)

    # Initialize
    >>> User(name="John")
    User(name='John', age=10)

    # Extra kwargs are accepted
    >>> User(name="John", age=30, lastname="Doe")
    User(name='John', age=30, *lastname='Doe')

    # Invalid type => raises
    >>> User(name="John", age="30")
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        TypeError: Field 'age' expected int, got str (value: '30')

    # Invalid value => raises
    >>> User(name="John", age=-1)
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        ValueError: Value must be positive, got -1
    ```
    """

    def wrap(cls: Type[T]) -> Type[T]:
        if not hasattr(cls, "__dataclass_fields__"):
            raise StrictDataclassDefinitionError(
                f"Class '{cls.__name__}' must be a dataclass before applying @strict."
            )

        # List and store validators
        field_validators: dict[str, list[Validator_T]] = {}
        for f in fields(cls):  # type: ignore
            validators = []
            validators.append(_create_type_validator(f))
            custom_validator = f.metadata.get("validator")
            if custom_validator is not None:
                if not isinstance(custom_validator, list):
                    custom_validator = [custom_validator]
                for validator in custom_validator:
                    if not _is_validator(validator):
                        raise StrictDataclassDefinitionError(
                            f"Invalid validator for field '{f.name}': {validator}. Must be a callable taking a single argument."
                        )
                validators.extend(custom_validator)
            field_validators[f.name] = validators
        cls.__validators__ = field_validators  # type: ignore

        # Override __setattr__ to validate fields on assignment
        original_setattr = cls.__setattr__

        def __strict_setattr__(self: Any, name: str, value: Any) -> None:
            """Custom __setattr__ method for strict dataclasses."""
            # Run all validators
            for validator in self.__validators__.get(name, []):
                try:
                    validator(value)
                except (ValueError, TypeError) as e:
                    raise StrictDataclassFieldValidationError(field=name, cause=e) from e

            # If validation passed, set the attribute
            original_setattr(self, name, value)

        cls.__setattr__ = __strict_setattr__  # type: ignore

        if accept_kwargs:
            # (optional) Override __init__ to accept arbitrary keyword arguments
            original_init = cls.__init__

            @wraps(original_init)
            def __init__(self, *args, **kwargs: Any) -> None:
                # Extract only the fields that are part of the dataclass
                dataclass_fields = {f.name for f in fields(cls)}  # type: ignore
                standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}

                # User shouldn't define custom `__init__` when `accepts_kwargs`, and instead
                # are advised to move field manipulation to `__post_init__` (e.g., derive new field from existing ones)
                # We need to call bare `__init__` here without `__post_init__` but the``original_init`` would call
                # post-init right away with no kwargs.
                if len(args) > 0:
                    raise ValueError(
                        f"When `accept_kwargs=True`, {cls.__name__} accepts only keyword arguments, "
                        f"but found `{len(args)}` positional args."
                    )

                for f in fields(cls):  # type: ignore
                    if f.name in standard_kwargs:
                        setattr(self, f.name, standard_kwargs[f.name])
                    elif f.default is not MISSING:
                        setattr(self, f.name, f.default)
                    elif f.default_factory is not MISSING:
                        setattr(self, f.name, f.default_factory())
                    else:
                        raise TypeError(f"Missing required field - '{f.name}'")

                # Pass any additional kwargs to `__post_init__` and let the object
                # decide whether to set the attr or use for different purposes (e.g. BC checks)
                additional_kwargs = {}
                for name, value in kwargs.items():
                    if name not in dataclass_fields:
                        additional_kwargs[name] = value

                self.__post_init__(**additional_kwargs)

            cls.__init__ = __init__  # type: ignore

            # Define a default __post_init__ if not defined
            if not hasattr(cls, "__post_init__"):

                def __post_init__(self, **kwargs: Any) -> None:
                    """Default __post_init__ to accept additional kwargs."""
                    for name, value in kwargs.items():
                        setattr(self, name, value)

                cls.__post_init__ = __post_init__  # type: ignore

            # (optional) Override __repr__ to include additional kwargs
            original_repr = cls.__repr__

            @wraps(original_repr)
            def __repr__(self) -> str:
                # Call the original __repr__ to get the standard fields
                standard_repr = original_repr(self)

                # Get additional kwargs
                additional_kwargs = [
                    # add a '*' in front of additional kwargs to let the user know they are not part of the dataclass
                    f"*{k}={v!r}"
                    for k, v in self.__dict__.items()
                    if k not in cls.__dataclass_fields__  # type: ignore [attr-defined]
                ]
                additional_repr = ", ".join(additional_kwargs)

                # Combine both representations
                return f"{standard_repr[:-1]}, {additional_repr})" if additional_kwargs else standard_repr

            if cls.__dataclass_params__.repr is True:  # type: ignore [attr-defined]
                cls.__repr__ = __repr__  # type: ignore

        # List all public methods starting with `validate_` => class validators.
        class_validators = []

        for name in dir(cls):
            if not name.startswith("validate_"):
                continue
            method = getattr(cls, name)
            if not callable(method):
                continue
            if len(inspect.signature(method).parameters) != 1:
                raise StrictDataclassDefinitionError(
                    f"Class '{cls.__name__}' has a class validator '{name}' that takes more than one argument."
                    " Class validators must take only 'self' as an argument. Methods starting with 'validate_'"
                    " are considered to be class validators."
                )
            class_validators.append(method)

        cls.__class_validators__ = class_validators  # type: ignore

        # Add `validate` method to the class, but first check if it already exists
        def validate(self: T) -> None:
            """Run class validators on the instance."""
            for validator in cls.__class_validators__:  # type: ignore [attr-defined]
                try:
                    validator(self)
                except (ValueError, TypeError) as e:
                    raise StrictDataclassClassValidationError(validator=validator.__name__, cause=e) from e

        # Hack to be able to raise if `.validate()` already exists except if it was created by this decorator on a parent class
        # (in which case we just override it)
        validate.__is_defined_by_strict_decorator__ = True  # type: ignore [attr-defined]

        if hasattr(cls, "validate"):
            if not getattr(cls.validate, "__is_defined_by_strict_decorator__", False):  # type: ignore [attr-defined]
                raise StrictDataclassDefinitionError(
                    f"Class '{cls.__name__}' already implements a method called 'validate'."
                    " This method name is reserved when using the @strict decorator on a dataclass."
                    " If you want to keep your own method, please rename it."
                )

        cls.validate = validate  # type: ignore

        # Run class validators after initialization
        initial_init = cls.__init__

        @wraps(initial_init)
        def init_with_validate(self, *args, **kwargs) -> None:
            """Run class validators after initialization."""
            initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
            cls.validate(self)  # type: ignore [attr-defined]

        setattr(cls, "__init__", init_with_validate)

        return cls

    # Return wrapped class or the decorator itself
    return wrap(cls) if cls is not None else wrap


def validate_typed_dict(schema: type[TypedDictType], data: dict) -> None:
    """
    Validate that a dictionary conforms to the types defined in a TypedDict class.

    Under the hood, the typed dict is converted to a strict dataclass and validated using the `@strict` decorator.

    Args:
        schema (`type[TypedDictType]`):
            The TypedDict class defining the expected structure and types.
        data (`dict`):
            The dictionary to validate.

    Raises:
        `StrictDataclassFieldValidationError`:
            If any field in the dictionary does not conform to the expected type.

    Example:
    ```py
    >>> from typing import Annotated, TypedDict
    >>> from huggingface_hub.dataclasses import validate_typed_dict

    >>> def positive_int(value: int):
    ...     if not value >= 0:
    ...         raise ValueError(f"Value must be positive, got {value}")

    >>> class User(TypedDict):
    ...     name: str
    ...     age: Annotated[int, positive_int]

    >>> # Valid data
    >>> validate_typed_dict(User, {"name": "John", "age": 30})

    >>> # Invalid type for age
    >>> validate_typed_dict(User, {"name": "John", "age": "30"})
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        TypeError: Field 'age' expected int, got str (value: '30')

    >>> # Invalid value for age
    >>> validate_typed_dict(User, {"name": "John", "age": -1})
    huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
        ValueError: Value must be positive, got -1
    ```
    """
    # Convert typed dict to dataclass
    strict_cls = _build_strict_cls_from_typed_dict(schema)

    # Validate the data by instantiating the strict dataclass
    strict_cls(**data)  # will raise if validation fails


@lru_cache
def _build_strict_cls_from_typed_dict(schema: type[TypedDictType]) -> Type:
    # Extract type hints from the TypedDict class
    type_hints = _get_typed_dict_annotations(schema)

    # If the TypedDict is not total, wrap fields as NotRequired (unless explicitly Required or NotRequired)
    if not getattr(schema, "__total__", True):
        for key, value in type_hints.items():
            origin = get_origin(value)

            if origin is Annotated:
                base, *meta = get_args(value)
                if not _is_required_or_notrequired(base):
                    base = NotRequired[base]
                type_hints[key] = Annotated[tuple([base] + list(meta))]  # type: ignore
            elif not _is_required_or_notrequired(value):
                type_hints[key] = NotRequired[value]

    # Convert type hints to dataclass fields
    fields = []
    for key, value in type_hints.items():
        if get_origin(value) is Annotated:
            base, *meta = get_args(value)
            fields.append((key, base, field(default=_TYPED_DICT_DEFAULT_VALUE, metadata={"validator": meta[0]})))
        else:
            fields.append((key, value, field(default=_TYPED_DICT_DEFAULT_VALUE)))

    # Create a strict dataclass from the TypedDict fields
    return strict(make_dataclass(schema.__name__, fields))


def _get_typed_dict_annotations(schema: type[TypedDictType]) -> dict[str, Any]:
    """Extract type annotations from a TypedDict class."""
    try:
        # Available in Python 3.14+
        import annotationlib

        return annotationlib.get_annotations(schema)
    except ImportError:
        return {
            # We do not use `get_type_hints` here to avoid evaluating ForwardRefs (which might fail).
            # ForwardRefs are not validated by @strict anyway.
            name: value if value is not None else type(None)
            for name, value in schema.__dict__.get("__annotations__", {}).items()
        }


def validated_field(
    validator: list[Validator_T] | Validator_T,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create a dataclass field with a custom validator.

    Useful to apply several checks to a field. If only applying one rule, check out the [`as_validated_field`] decorator.

    Args:
        validator (`Callable` or `list[Callable]`):
            A method that takes a value as input and raises ValueError/TypeError if the value is invalid.
            Can be a list of validators to apply multiple checks.
        **kwargs:
            Additional arguments to pass to `dataclasses.field()`.

    Returns:
        A field with the validator attached in metadata
    """
    if not isinstance(validator, list):
        validator = [validator]
    if metadata is None:
        metadata = {}
    metadata["validator"] = validator
    return field(  # type: ignore
        default=default,  # type: ignore
        default_factory=default_factory,  # type: ignore
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        **kwargs,
    )


def as_validated_field(validator: Validator_T):
    """
    Decorates a validator function as a [`validated_field`] (i.e. a dataclass field with a custom validator).

    Args:
        validator (`Callable`):
            A method that takes a value as input and raises ValueError/TypeError if the value is invalid.
    """

    def _inner(
        default: Any = MISSING,
        default_factory: Any = MISSING,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        return validated_field(
            validator,
            default=default,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata,
            **kwargs,
        )

    return _inner


def type_validator(name: str, value: Any, expected_type: Any) -> None:
    """Validate that 'value' matches 'expected_type'."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return
    elif expected_type is None:
        _validate_none(name, value)
    elif validator := _BASIC_TYPE_VALIDATORS.get(origin):
        validator(name, value, args)
    elif isinstance(expected_type, type):  # simple types
        _validate_simple_type(name, value, expected_type)
    elif isinstance(expected_type, ForwardRef) or isinstance(expected_type, str):
        return
    elif origin is Required:
        if value is _TYPED_DICT_DEFAULT_VALUE:
            raise TypeError(f"Field '{name}' is required but missing.")
        type_validator(name, value, args[0])
    elif origin is NotRequired:
        if value is _TYPED_DICT_DEFAULT_VALUE:
            return
        type_validator(name, value, args[0])
    else:
        raise TypeError(f"Unsupported type for field '{name}': {expected_type}")


def _validate_none(name: str, value: Any) -> None:
    """Validate None type.

    'None' is not a type, it's a special value. Type should be `NoneType` instead.
    But in type annotations 'None' is accepted so we must support it.
    """
    if value is not None:
        raise TypeError(f"Field '{name}' expected None, got {type(value).__name__}")


def _validate_union(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate that value matches one of the types in a Union."""
    errors = []
    for t in args:
        try:
            type_validator(name, value, t)
            return  # Valid if any type matches
        except TypeError as e:
            errors.append(str(e))

    raise TypeError(
        f"Field '{name}' with value {repr(value)} doesn't match any type in {args}. Errors: {'; '.join(errors)}"
    )


def _validate_literal(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Literal type."""
    if isinstance(value, bool):
        if value not in [arg for arg in args if isinstance(arg, bool)]:
            raise TypeError(f"Field '{name}' expected one of {args}, got {value}")
    elif isinstance(value, int):
        if value not in [arg for arg in args if isinstance(arg, int) and not isinstance(arg, bool)]:
            raise TypeError(f"Field '{name}' expected one of {args}, got {value}")
    elif value not in args:
        raise TypeError(f"Field '{name}' expected one of {args}, got {value}")


def _validate_list(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate list or list[T] type."""
    if not isinstance(value, list):
        raise TypeError(f"Field '{name}' expected a list, got {type(value).__name__}")

    # If no type argument is provided (i.e., bare `typing.List`), skip item validation
    if not args:
        return

    # Validate each item in the list
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in list '{name}'") from e


def _validate_dict(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate dict or dict[K, V] type."""
    if not isinstance(value, dict):
        raise TypeError(f"Field '{name}' expected a dict, got {type(value).__name__}")

    # If no type arguments are provided (i.e., bare `typing.Dict`), skip key/value validation
    if not args:
        return

    # Validate keys and values
    key_type, value_type = args
    for k, v in value.items():
        try:
            type_validator(f"{name}.key", k, key_type)
            type_validator(f"{name}[{k!r}]", v, value_type)
        except TypeError as e:
            raise TypeError(f"Invalid key or value in dict '{name}'") from e


def _validate_tuple(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Tuple type."""
    if not isinstance(value, tuple):
        raise TypeError(f"Field '{name}' expected a tuple, got {type(value).__name__}")

    # Handle variable-length tuples: tuple[T, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        for i, item in enumerate(value):
            try:
                type_validator(f"{name}[{i}]", item, args[0])
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e
    # Handle fixed-length tuples: tuple[T1, T2, ...]
    elif len(args) != len(value):
        raise TypeError(f"Field '{name}' expected a tuple of length {len(args)}, got {len(value)}")
    else:
        for i, (item, expected) in enumerate(zip(value, args)):
            try:
                type_validator(f"{name}[{i}]", item, expected)
            except TypeError as e:
                raise TypeError(f"Invalid item at index {i} in tuple '{name}'") from e


def _validate_set(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate set or set[T] type."""
    if not isinstance(value, set):
        raise TypeError(f"Field '{name}' expected a set, got {type(value).__name__}")

    # If no type argument is provided (i.e., bare `typing.Set`), skip item validation
    if not args:
        return

    # Validate each item in the set
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name} item", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item in set '{name}'") from e


def _validate_sequence(name: str, value: Any, args: tuple[Any, ...]) -> None:
    """Validate Sequence or Sequence[T] type."""
    if not isinstance(value, collections.abc.Sequence):
        raise TypeError(f"Field '{name}' expected a Sequence, got {type(value).__name__}")

    # If no type argument is provided (i.e., just `Sequence`), skip item validation
    if not args:
        return

    # Validate each item in the sequence
    item_type = args[0]
    for i, item in enumerate(value):
        try:
            type_validator(f"{name}[{i}]", item, item_type)
        except TypeError as e:
            raise TypeError(f"Invalid item at index {i} in sequence '{name}'") from e


def _validate_simple_type(name: str, value: Any, expected_type: type) -> None:
    """Validate simple type (int, str, etc.)."""
    if expected_type is int and isinstance(value, bool):
        raise TypeError(
            f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
        )
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
        )


def _create_type_validator(field: Field) -> Validator_T:
    """Create a type validator function for a field."""
    # Hacky: we cannot use a lambda here because of reference issues

    def validator(value: Any) -> None:
        type_validator(field.name, value, field.type)

    return validator


def _is_validator(validator: Any) -> bool:
    """Check if a function is a validator.

    A validator is a Callable that can be called with a single positional argument.
    The validator can have more arguments with default values.

    Basically, returns True if `validator(value)` is possible.
    """
    if not callable(validator):
        return False

    signature = inspect.signature(validator)
    parameters = list(signature.parameters.values())
    if len(parameters) == 0:
        return False
    if parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        return False
    for parameter in parameters[1:]:
        if parameter.default == inspect.Parameter.empty:
            return False
    return True


def _is_required_or_notrequired(type_hint: Any) -> bool:
    """Helper to check if a type is Required/NotRequired."""
    return type_hint in (Required, NotRequired) or (get_origin(type_hint) in (Required, NotRequired))


_BASIC_TYPE_VALIDATORS: dict[Any, Callable[[str, Any, tuple[Any, ...]], None]] = {
    Union: _validate_union,
    Literal: _validate_literal,
    list: _validate_list,
    dict: _validate_dict,
    tuple: _validate_tuple,
    set: _validate_set,
    collections.abc.Sequence: _validate_sequence,
}

# TODO: make it first class citizen when bumping to Python 3.10+
_BASIC_TYPE_VALIDATORS[types.UnionType] = _validate_union  # x | y syntax, available only Python 3.10+


__all__ = [
    "strict",
    "validate_typed_dict",
    "validated_field",
    "Validator_T",
    "StrictDataclassClassValidationError",
    "StrictDataclassDefinitionError",
    "StrictDataclassFieldValidationError",
]
<|endoftext|>from . import constants


class ResolvedRevision(str):
    """A git revision that has already been resolved to a commit hash.

    `ResolvedRevision` is a `str` subclass, so it can be passed to any `huggingface_hub` method taking a `revision`
    argument. Its string value is the revision initially requested by the user (e.g. `"main"`, `"refs/pr/4"`),
    which keeps URLs and error messages readable, while `.resolved` holds the commit hash it points to.

    Instances are built by [`HfApi.resolve_revision`], which also caches the `revision` -> `commit hash` mapping
    in the local cache (`refs/` folder).

    A commit hash only means something for the repo it was resolved against, so an instance also remembers that
    repo. Re-resolving it for another repo is not an error: the revision initially requested is resolved again
    (see [`HfApi.resolve_revision`]).

    Attributes:
        initial (`str` or `None`):
            The revision initially requested by the user. If `None`, the string value defaults to `"main"`.
        resolved (`str`):
            The commit hash that `initial` resolves to.

    Example:
    ```python
    >>> from huggingface_hub import resolve_revision
    >>> revision = resolve_revision("openai-community/gpt2")
    >>> revision
    ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    >>> revision == "main"  # it's a string
    True
    >>> revision.resolved
    '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    ```
    """

    initial: str | None
    resolved: str
    _repo_id: str | None
    _repo_type: str

    def __new__(
        cls,
        resolved: str,
        initial: str | None = None,
        repo_id: str | None = None,
        repo_type: str | None = None,
    ) -> "ResolvedRevision":
        revision = super().__new__(cls, initial if initial is not None else constants.DEFAULT_REVISION)
        revision.initial = initial
        revision.resolved = resolved
        # The repo `resolved` belongs to. `None` means unknown, in which case it is assumed to fit any repo.
        revision._repo_id = repo_id
        revision._repo_type = repo_type or constants.REPO_TYPE_MODEL
        return revision

    def __reduce__(self):
        # without this, pickle/copy rebuild the instance from its string value only, losing the attributes
        return self.__class__, (self.resolved, self.initial, self._repo_id, self._repo_type)

    def __repr__(self) -> str:
        return f"ResolvedRevision(initial={self.initial!r}, resolved={self.resolved!r})"
<|endoftext|>import inspect
import json
import os
from collections.abc import Callable
from dataclasses import Field, asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast

import packaging.version

from . import constants
from .errors import EntryNotFoundError, HfHubHTTPError
from .file_download import hf_hub_download
from .hf_api import HfApi
from .repocard import ModelCard, ModelCardData
from .utils import (
    SoftTemporaryDirectory,
    is_jsonable,
    is_simple_optional_type,
    logging,
    unwrap_simple_optional_type,
    validate_hf_hub_args,
)


logger = logging.get_logger(__name__)


# Type alias for dataclass instances, copied from https://github.com/python/typeshed/blob/9f28171658b9ca6c32a7cb93fbb99fc92b17858b/stdlib/_typeshed/__init__.pyi#L349
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field]]


# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="ModelHubMixin")
# Generic variable to represent an args type
ARGS_T = TypeVar("ARGS_T")
ENCODER_T = Callable[[ARGS_T], Any]
DECODER_T = Callable[[Any], ARGS_T]
CODER_T = tuple[ENCODER_T, DECODER_T]


DEFAULT_MODEL_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: {{ repo_url | default("[More Information Needed]", true) }}
- Paper: {{ paper_url | default("[More Information Needed]", true) }}
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


@dataclass
class MixinInfo:
    model_card_template: str
    model_card_data: ModelCardData
    docs_url: str | None = None
    paper_url: str | None = None
    repo_url: str | None = None


class ModelHubMixin:
    """
    A generic mixin to integrate ANY machine learning framework with the Hub.

    To integrate your framework, your model class must inherit from this class. Custom logic for saving/loading models
    have to be overwritten in  [`_from_pretrained`] and [`_save_pretrained`]. [`PyTorchModelHubMixin`] is a good example
    of mixin integration with the Hub. Check out our [integration guide](../guides/integrations) for more instructions.

    When inheriting from [`ModelHubMixin`], you can define class-level attributes. These attributes are not passed to
    `__init__` but to the class definition itself. This is useful to define metadata about the library integrating
    [`ModelHubMixin`].

    For more details on how to integrate the mixin with your library, checkout the [integration guide](../guides/integrations).

    Args:
        repo_url (`str`, *optional*):
            URL of the library repository. Used to generate model card.
        paper_url (`str`, *optional*):
            URL of the library paper. Used to generate model card.
        docs_url (`str`, *optional*):
            URL of the library documentation. Used to generate model card.
        model_card_template (`str`, *optional*):
            Template of the model card. Used to generate model card. Defaults to a generic template.
        language (`str` or `list[str]`, *optional*):
            Language supported by the library. Used to generate model card.
        library_name (`str`, *optional*):
            Name of the library integrating ModelHubMixin. Used to generate model card.
        license (`str`, *optional*):
            License of the library integrating ModelHubMixin. Used to generate model card.
            E.g: "apache-2.0"
        license_name (`str`, *optional*):
            Name of the library integrating ModelHubMixin. Used to generate model card.
            Only used if `license` is set to `other`.
            E.g: "coqui-public-model-license".
        license_link (`str`, *optional*):
            URL to the license of the library integrating ModelHubMixin. Used to generate model card.
            Only used if `license` is set to `other` and `license_name` is set.
            E.g: "https://coqui.ai/cpml".
        pipeline_tag (`str`, *optional*):
            Tag of the pipeline. Used to generate model card. E.g. "text-classification".
        tags (`list[str]`, *optional*):
            Tags to be added to the model card. Used to generate model card. E.g. ["computer-vision"]
        coders (`dict[Type, tuple[Callable, Callable]]`, *optional*):
            Dictionary of custom types and their encoders/decoders. Used to encode/decode arguments that are not
            jsonable by default. E.g. dataclasses, argparse.Namespace, OmegaConf, etc.

    Example:

    ```python
    >>> from huggingface_hub import ModelHubMixin

    # Inherit from ModelHubMixin
    >>> class MyCustomModel(
    ...         ModelHubMixin,
    ...         library_name="my-library",
    ...         tags=["computer-vision"],
    ...         repo_url="https://github.com/huggingface/my-cool-library",
    ...         paper_url="https://arxiv.org/abs/2304.12244",
    ...         docs_url="https://huggingface.co/docs/my-cool-library",
    ...         # ^ optional metadata to generate model card
    ...     ):
    ...     def __init__(self, size: int = 512, device: str = "cpu"):
    ...         # define how to initialize your model
    ...         super().__init__()
    ...         ...
    ...
    ...     def _save_pretrained(self, save_directory: Path) -> None:
    ...         # define how to serialize your model
    ...         ...
    ...
    ...     @classmethod
    ...     def from_pretrained(
    ...         cls: type[T],
    ...         pretrained_model_name_or_path: Union[str, Path],
    ...         *,
    ...         force_download: bool = False,
    ...         token: Optional[Union[str, bool]] = None,
    ...         cache_dir: Optional[Union[str, Path]] = None,
    ...         local_files_only: bool = False,
    ...         revision: Optional[str] = None,
    ...         **model_kwargs,
    ...     ) -> T:
    ...         # define how to deserialize your model
    ...         ...

    >>> model = MyCustomModel(size=256, device="gpu")

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> reloaded_model = MyCustomModel.from_pretrained("username/my-awesome-model")
    >>> reloaded_model.size
    256

    # Model card has been correctly populated
    >>> from huggingface_hub import ModelCard
    >>> card = ModelCard.load("username/my-awesome-model")
    >>> card.data.tags
    ["x-custom-tag", "pytorch_model_hub_mixin", "model_hub_mixin"]
    >>> card.data.library_name
    "my-library"
    ```
    """

    _hub_mixin_config: dict | DataclassInstance | None = None
    # ^ optional config attribute automatically set in `from_pretrained`
    _hub_mixin_info: MixinInfo
    # ^ information about the library integrating ModelHubMixin (used to generate model card)
    _hub_mixin_inject_config: bool  # whether `_from_pretrained` expects `config` or not
    _hub_mixin_init_parameters: dict[str, inspect.Parameter]  # __init__ parameters
    _hub_mixin_jsonable_default_values: dict[str, Any]  # default values for __init__ parameters
    _hub_mixin_jsonable_custom_types: tuple[type, ...]  # custom types that can be encoded/decoded
    _hub_mixin_coders: dict[type, CODER_T]  # encoders/decoders for custom types
    # ^ internal values to handle config

    def __init_subclass__(
        cls,
        *,
        # Generic info for model card
        repo_url: str | None = None,
        paper_url: str | None = None,
        docs_url: str | None = None,
        # Model card template
        model_card_template: str = DEFAULT_MODEL_CARD,
        # Model card metadata
        language: list[str] | None = None,
        library_name: str | None = None,
        license: str | None = None,
        license_name: str | None = None,
        license_link: str | None = None,
        pipeline_tag: str | None = None,
        tags: list[str] | None = None,
        # How to encode/decode arguments with custom type into a JSON config?
        coders: None
        | (
            dict[type, CODER_T]
            # Key is a type.
            # Value is a tuple (encoder, decoder).
            # Example: {MyCustomType: (lambda x: x.value, lambda data: MyCustomType(data))}
        ) = None,
    ) -> None:
        """Inspect __init__ signature only once when subclassing + handle modelcard."""
        super().__init_subclass__()

        # Will be reused when creating modelcard
        tags = tags or []
        tags.append("model_hub_mixin")

        # Initialize MixinInfo if not existent
        info = MixinInfo(model_card_template=model_card_template, model_card_data=ModelCardData())

        # If parent class has a MixinInfo, inherit from it as a copy
        if hasattr(cls, "_hub_mixin_info"):
            # Inherit model card template from parent class if not explicitly set
            if model_card_template == DEFAULT_MODEL_CARD:
                info.model_card_template = cls._hub_mixin_info.model_card_template

            # Inherit from parent model card data
            info.model_card_data = ModelCardData(**cls._hub_mixin_info.model_card_data.to_dict())

            # Inherit other info
            info.docs_url = cls._hub_mixin_info.docs_url
            info.paper_url = cls._hub_mixin_info.paper_url
            info.repo_url = cls._hub_mixin_info.repo_url
        cls._hub_mixin_info = info

        # Update MixinInfo with metadata
        if model_card_template is not None and model_card_template != DEFAULT_MODEL_CARD:
            info.model_card_template = model_card_template
        if repo_url is not None:
            info.repo_url = repo_url
        if paper_url is not None:
            info.paper_url = paper_url
        if docs_url is not None:
            info.docs_url = docs_url
        if language is not None:
            info.model_card_data.language = language
        if library_name is not None:
            info.model_card_data.library_name = library_name
        if license is not None:
            info.model_card_data.license = license
        if license_name is not None:
            info.model_card_data.license_name = license_name
        if license_link is not None:
            info.model_card_data.license_link = license_link
        if pipeline_tag is not None:
            info.model_card_data.pipeline_tag = pipeline_tag
        if tags is not None:
            normalized_tags = list(tags)
            if info.model_card_data.tags is not None:
                info.model_card_data.tags.extend(normalized_tags)
            else:
                info.model_card_data.tags = normalized_tags

        if info.model_card_data.tags is not None:
            info.model_card_data.tags = sorted(set(info.model_card_data.tags))

        # Handle encoders/decoders for args
        cls._hub_mixin_coders = coders or {}
        cls._hub_mixin_jsonable_custom_types = tuple(cls._hub_mixin_coders.keys())

        # Inspect __init__ signature to handle config
        cls._hub_mixin_init_parameters = dict(inspect.signature(cls.__init__).parameters)
        cls._hub_mixin_jsonable_default_values = {
            param.name: cls._encode_arg(param.default)
            for param in cls._hub_mixin_init_parameters.values()
            if param.default is not inspect.Parameter.empty and cls._is_jsonable(param.default)
        }
        cls._hub_mixin_inject_config = "config" in inspect.signature(cls._from_pretrained).parameters

    def __new__(cls: type[T], *args, **kwargs) -> T:
        """Create a new instance of the class and handle config.

        3 cases:
        - If `self._hub_mixin_config` is already set, do nothing.
        - If `config` is passed as a dataclass, set it as `self._hub_mixin_config`.
        - Otherwise, build `self._hub_mixin_config` from default values and passed values.
        """
        instance = super().__new__(cls)

        # If `config` is already set, return early
        if instance._hub_mixin_config is not None:
            return instance

        # Infer passed values
        passed_values = {
            **{
                key: value
                for key, value in zip(
                    # [1:] to skip `self` parameter
                    list(cls._hub_mixin_init_parameters)[1:],
                    args,
                )
            },
            **kwargs,
        }

        # If config passed as dataclass => set it and return early
        if is_dataclass(passed_values.get("config")):
            instance._hub_mixin_config = passed_values["config"]
            return instance

        # Otherwise, build config from default + passed values
        init_config = {
            # default values
            **cls._hub_mixin_jsonable_default_values,
            # passed values
            **{
                key: cls._encode_arg(value)  # Encode custom types as jsonable value
                for key, value in passed_values.items()
                if instance._is_jsonable(value)  # Only if jsonable or we have a custom encoder
            },
        }
        passed_config = init_config.pop("config", {})

        # Populate `init_config` with provided config
        if isinstance(passed_config, dict):
            init_config.update(passed_config)

        # Set `config` attribute and return
        if init_config != {}:
            instance._hub_mixin_config = init_config
        return instance

    @classmethod
    def _is_jsonable(cls, value: Any) -> bool:
        """Check if a value is JSON serializable."""
        if is_dataclass(value):
            return True
        if isinstance(value, cls._hub_mixin_jsonable_custom_types):
            return True
        return is_jsonable(value)

    @classmethod
    def _encode_arg(cls, arg: Any) -> Any:
        """Encode an argument into a JSON serializable format."""
        if is_dataclass(arg):
            return asdict(arg)  # type: ignore[arg-type]
        for type_, (encoder, _) in cls._hub_mixin_coders.items():
            if isinstance(arg, type_):
                if arg is None:
                    return None
                return encoder(arg)
        return arg

    @classmethod
    def _decode_arg(cls, expected_type: type[ARGS_T], value: Any) -> ARGS_T | None:
        """Decode a JSON serializable value into an argument."""
        if is_simple_optional_type(expected_type):
            if value is None:
                return None
            expected_type = unwrap_simple_optional_type(expected_type)  # type: ignore
        # Dataclass => handle it
        if is_dataclass(expected_type):
            return _load_dataclass(expected_type, value)  # type: ignore
        # Otherwise => check custom decoders
        for type_, (_, decoder) in cls._hub_mixin_coders.items():
            if inspect.isclass(expected_type) and issubclass(expected_type, type_):
                return decoder(value)
        # Otherwise => don't decode
        return value

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict | DataclassInstance | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            model_card_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments passed to the model card template to customize the model card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Remove config.json if already exists. After `_save_pretrained` we don't want to overwrite config.json
        # as it might have been saved by the custom `_save_pretrained` already. However we do want to overwrite
        # an existing config.json if it was not saved by `_save_pretrained`.
        config_path = save_directory / constants.CONFIG_NAME
        config_path.unlink(missing_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # save config (if provided and if not serialized yet in `_save_pretrained`)
        if config is None:
            config = self._hub_mixin_config
        if config is not None:
            if is_dataclass(config):
                config = asdict(config)  # type: ignore[arg-type]
            if not config_path.exists():
                config_str = json.dumps(config, sort_keys=True, indent=2)
                config_path.write_text(config_str)

        # save model card
        model_card_path = save_directory / "README.md"
        model_card_kwargs = model_card_kwargs if model_card_kwargs is not None else {}
        if not model_card_path.exists():  # do not overwrite if already exists
            self.generate_model_card(**model_card_kwargs).save(save_directory / "README.md")

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, model_card_kwargs=model_card_kwargs, **kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your model.
        Check out our [integration guide](../guides/integrations) for instructions.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: type[T],
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs,
    ) -> T:
        """
        Download a model from the Huggingface Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `hf auth login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs (`dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = str(pretrained_model_name_or_path)
        config_file: str | None = None
        if os.path.isdir(model_id):
            if constants.CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, constants.CONFIG_NAME)
            else:
                logger.warning(f"{constants.CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                logger.info(f"{constants.CONFIG_NAME} not found on the HuggingFace Hub: {str(e)}")

        # Read config
        config = None
        if config_file is not None:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Decode custom types in config
            for key, value in config.items():
                if key in cls._hub_mixin_init_parameters:
                    expected_type = cls._hub_mixin_init_parameters[key].annotation
                    if expected_type is not inspect.Parameter.empty:
                        config[key] = cls._decode_arg(expected_type, value)

            # Populate model_kwargs from config
            for param in cls._hub_mixin_init_parameters.values():
                if param.name not in model_kwargs and param.name in config:
                    model_kwargs[param.name] = config[param.name]

            # Check if `config` argument was passed at init
            if "config" in cls._hub_mixin_init_parameters and "config" not in model_kwargs:
                # Decode `config` argument if it was passed
                config_annotation = cls._hub_mixin_init_parameters["config"].annotation
                config = cls._decode_arg(config_annotation, config)

                # Forward config to model initialization
                model_kwargs["config"] = config

            # Inject config if `**kwargs` are expected
            if is_dataclass(cls):
                for key in cls.__dataclass_fields__:
                    if key not in model_kwargs and key in config:
                        model_kwargs[key] = config[key]
            elif any(param.kind == inspect.Parameter.VAR_KEYWORD for param in cls._hub_mixin_init_parameters.values()):
                for key, value in config.items():  # type: ignore[union-attr]
                    if key not in model_kwargs:
                        model_kwargs[key] = value

            # Finally, also inject if `_from_pretrained` expects it
            if cls._hub_mixin_inject_config and "config" not in model_kwargs:
                model_kwargs["config"] = config

        instance = cls._from_pretrained(
            model_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        # Implicitly set the config as instance attribute if not already set by the class
        # This way `config` will be available when calling `save_pretrained` or `push_to_hub`.
        if config is not None and (getattr(instance, "_hub_mixin_config", None) in (None, {})):
            instance._hub_mixin_config = config

        return instance

    @classmethod
    def _from_pretrained(
        cls: type[T],
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs,
    ) -> T:
        """Overwrite this method in subclass to define how to load your model from pretrained.

        Use [`hf_hub_download`] or [`snapshot_download`] to download files from the Hub before loading them. Most
        args taken as input can be directly passed to those 2 methods. If needed, you can add more arguments to this
        method using "model_kwargs". For example [`PyTorchModelHubMixin._from_pretrained`] takes as input a `map_location`
        parameter to set on which device the model should be loaded.

        Check out our [integration guide](../guides/integrations) for more instructions.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `hf auth login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DataclassInstance | None = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether the repository created should be private.
                If `None` (default), the repo will be public unless the organization's default is private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `hf auth login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`list[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`list[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`list[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.
            model_card_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments passed to the model card template to customize the model card.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config, model_card_kwargs=model_card_kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        card = ModelCard.from_template(
            card_data=self._hub_mixin_info.model_card_data,
            template_str=self._hub_mixin_info.model_card_template,
            repo_url=self._hub_mixin_info.repo_url,
            paper_url=self._hub_mixin_info.paper_url,
            docs_url=self._hub_mixin_info.docs_url,
            **kwargs,
        )
        return card


class PyTorchModelHubMixin(ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download capabilities to PyTorch models. The model
    is set in evaluation mode by default using `model.eval()` (dropout modules are deactivated). To train the model,
    you should first set it back in training mode with `model.train()`.

    See [`ModelHubMixin`] for more details on how to use the mixin.

    Example:

    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> from huggingface_hub import PyTorchModelHubMixin

    >>> class MyModel(
    ...         nn.Module,
    ...         PyTorchModelHubMixin,
    ...         library_name="keras-nlp",
    ...         repo_url="https://github.com/keras-team/keras-nlp",
    ...         paper_url="https://arxiv.org/abs/2304.12244",
    ...         docs_url="https://keras.io/keras_nlp/",
    ...         # ^ optional metadata to generate model card
    ...     ):
    ...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
    ...         self.linear = nn.Linear(output_size, vocab_size)

    ...     def forward(self, x):
    ...         return self.linear(x + self.param)
    >>> model = MyModel(hidden_size=256)

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    >>> model.hidden_size
    256
    ```
    """

    def __init_subclass__(cls, *args, tags: list[str] | None = None, **kwargs) -> None:
        tags = tags or []
        tags.append("pytorch_model_hub_mixin")
        kwargs["tags"] = tags
        return super().__init_subclass__(*args, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        from safetensors.torch import save_model as save_model_as_safetensor

        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(cast(Any, model_to_save), str(save_directory / constants.SAFETENSORS_SINGLE_FILE))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)

    @classmethod
    def _load_as_pickle(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        import torch

        state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore
        return model

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        import safetensors
        import safetensors.torch

        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):  # type: ignore [attr-defined]
            safetensors.torch.load_model(cast(Any, model), model_file, strict=strict)
            if map_location != "cpu":
                logger.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)  # type: ignore [attr-defined]
        else:
            safetensors.torch.load_model(cast(Any, model), model_file, strict=strict, device=map_location)
        model.eval()  # type: ignore
        return model


def _load_dataclass(datacls: type[DataclassInstance], data: dict) -> DataclassInstance:
    """Load a dataclass instance from a dictionary.

    Fields not expected by the dataclass are ignored.
    """
    return datacls(**{k: v for k, v in data.items() if k in datacls.__dataclass_fields__})
<|endoftext|>import os
from pathlib import Path
from typing import Literal, overload

import httpx2
from tqdm.auto import tqdm as base_tqdm

from . import constants
from ._revision import ResolvedRevision
from ._tree_cache import TreeCacheEntry, read_tree_cache, tree_cache_folder_for_local_dir, write_tree_cache
from .errors import (
    CachedRepoTreeNotFoundError,
    DryRunError,
    GatedRepoError,
    HfHubHTTPError,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from .file_download import (
    REGEX_COMMIT_HASH,
    DryRunFileInfo,
    _cache_commit_hash_for_specific_revision,
    hf_hub_download,
    repo_folder_name,
)
from .hf_api import HfApi, RepoFile
from .utils import OfflineModeIsEnabled, filter_repo_objects, logging, validate_hf_hub_args
from .utils._http import flag_as_download_call
from .utils._paths import as_extended_path
from .utils._xet_progress_reporting import (
    XET_BYTES_BAR_FORMAT,
    XET_TRANSFER_BAR_FORMAT,
    _finish_transfer_bar,
    _set_aggregate_rate_postfix,
    _update_transfer_bar,
)
from .utils.tqdm import _create_progress_bar, hf_thread_map
from .utils.tqdm import tqdm as hf_tqdm


logger = logging.get_logger(__name__)


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: Literal[False] = False,
) -> str: ...


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: Literal[True] = True,
) -> list[DryRunFileInfo]: ...


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: bool = False,
) -> str | list[DryRunFileInfo]: ...


@validate_hf_hub_args
@flag_as_download_call
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: bool = False,
) -> str | list[DryRunFileInfo]:
    """Download repo files.

    Download a whole snapshot of a repo's files at the specified revision. This is useful when you want all files from
    a repo because you don't know which ones you will need _a priori_. All files are nested in a folder to keep their
    path and filename relative to that folder. You can also filter which files to download by using `allow_patterns`
    and `ignore_patterns`.

    If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this
    option, the `cache_dir` will not be used, and a `.cache/huggingface/` folder will be created at the root of `local_dir`
    to store some metadata related to the downloaded files. While this mechanism is not as robust as the main
    cache system, it's optimized for regularly pulling the latest version of a repository.

    An alternative would be to clone the repo, but this requires git and git-lfs to be installed and properly
    configured. It is also not possible to filter which files to download when cloning a repository using git.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        repo_type (`str`, *optional*):
            Set to `"dataset"`, `"space"` or `"kernel"` if downloading from a dataset, space or kernel repo,
            `None` or `"model"` if downloading from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id, which can be a branch name, a tag, or a
            commit hash.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        local_dir (`str` or `Path`, *optional*):
            If provided, the downloaded files will be placed under this directory.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`str`, `dict`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        etag_timeout (`float`, *optional*, defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up, which is passed to `httpx2.request`.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in the local cache.
        token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        headers (`dict`, *optional*):
            Additional headers to include in the request. Those headers take precedence over the others.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, do not download any files even if they are not in `cache_dir` or `local_dir`.
        allow_patterns (`list[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are downloaded.
        ignore_patterns (`list[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not downloaded.
        max_workers (`int`, *optional*):
            Number of concurrent threads to download files (1 thread = 1 file download).
            Defaults to 8.
        tqdm_class (`tqdm`, *optional*):
            If provided, overwrites the default behavior for the progress bar. Passed
            argument must inherit from `tqdm.auto.tqdm` or at least mimic its behavior.
            Note that the `tqdm_class` is not passed to each individual download.
            Defaults to the custom HF progress bar that can be disabled by setting
            `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.
        dry_run (`bool`, *optional*, defaults to `False`):
            If `True`, perform a dry run without actually downloading the files. Returns a list of
            [`DryRunFileInfo`] objects containing information about what would be downloaded.

    Returns:
        `str` or list of [`DryRunFileInfo`]:
            - If `dry_run=False`: Local snapshot path.
            - If `dry_run=True`: A list of [`DryRunFileInfo`] objects containing download information.

    Raises:
        [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist
            or because it is set to `private` and you do not have access.
        [`~utils.RevisionNotFoundError`]
            If the revision to download from cannot be found.
        [`~errors.IncompleteSnapshotError`]
            If the Hub cannot be reached (offline, connection issue, or `local_files_only=True`) and the
            cached snapshot is missing some of the requested files.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `token=True` and the token cannot be found.
        [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
            ETag cannot be determined.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If some parameter value is invalid.
    """
    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())
    if local_dir is not None:
        local_dir = str(Path(local_dir).expanduser().resolve())
    if revision is None:
        revision = constants.DEFAULT_REVISION

    if repo_type is None:
        repo_type = "model"
    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES_WITH_KERNEL)}"
        )

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    # Folder under which the per-commit tree listing (`trees/<commit_hash>.json`) is cached on disk.
    tree_cache_folder = tree_cache_folder_for_local_dir(local_dir) if local_dir is not None else storage_folder

    api = HfApi(
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        endpoint=endpoint,
        headers=headers,
        token=token,
    )

    # The revision is already a commit hash if:
    # - it's a `ResolvedRevision` (already resolved, see [`HfApi.resolve_revision`])
    # - it's a commit hash, which is immutable
    # => in both cases, there is nothing to resolve and the `repo_info` call can be skipped.
    commit_hash: str | None = None
    if isinstance(revision, ResolvedRevision):
        commit_hash = revision.resolved
    elif REGEX_COMMIT_HASH.fullmatch(revision):
        commit_hash = revision

    tree_entries: dict[str, TreeCacheEntry] | None = None
    api_call_error: Exception | None = None
    if not local_files_only:
        # try/except logic to handle different errors => taken from `hf_hub_download`
        try:
            if commit_hash is None:
                # if we have internet connection we want to list files to download
                repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
                assert repo_info.sha is not None, "Repo info returned from server must have a revision sha."
                commit_hash = repo_info.sha
            tree_entries = read_tree_cache(tree_cache_folder, commit_hash)
            if tree_entries is None:
                tree_entries = {
                    f.path: TreeCacheEntry(
                        size=f.size,
                        blob_id=f.blob_id,
                        lfs_sha256=f.lfs.sha256 if f.lfs is not None else None,
                        lfs_size=f.lfs.size if f.lfs is not None else None,
                        xet_hash=f.xet_hash,
                    )
                    for f in api.list_repo_tree(
                        repo_id=repo_id, recursive=True, revision=commit_hash, repo_type=repo_type
                    )
                    if isinstance(f, RepoFile)
                }
                if not dry_run:
                    write_tree_cache(tree_cache_folder, commit_hash, tree_entries)
        except httpx2.ProxyError:
            # Actually raise on proxy error
            raise
        except (httpx2.ConnectError, httpx2.TimeoutException, OfflineModeIsEnabled) as error:
            # Internet connection is down
            # => will try to use local files only
            api_call_error = error
            pass
        except RevisionNotFoundError:
            # The repo was found but the revision doesn't exist on the Hub (never existed or got deleted)
            raise
        except HfHubHTTPError as error:
            # Multiple reasons for an http error:
            # - Repository is private and invalid/missing token sent
            # - Repository is gated and invalid/missing token sent
            # - Hub is down (error 500 or 504)
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            api_call_error = error
            pass

    # At this stage, if the tree listing is unknown it means either:
    # - internet connection is down
    # - internet connection is deactivated (local_files_only=True or HF_HUB_OFFLINE=True)
    # - repo is private/gated and invalid/missing token sent
    # - Hub is down
    # => let's look if we can find the appropriate folder in the cache:
    #    - if the specified revision is a commit hash, look inside "snapshots".
    #    - f the specified revision is a branch or tag, look inside "refs".
    # => if local_dir is not None, we will return the path to the local folder if it exists.
    if tree_entries is None or local_files_only:
        if dry_run:
            raise DryRunError(
                "Dry run cannot be performed as the repository cannot be accessed. Please check your internet connection or authentication token."
            ) from api_call_error

        # Try to get which commit hash corresponds to the specified revision
        if commit_hash is None:
            ref_path = os.path.join(storage_folder, "refs", revision)
            if os.path.exists(ref_path):
                # retrieve commit_hash from refs file
                with open(ref_path) as f:
                    commit_hash = f.read()

        # Try to locate snapshot folder for this commit hash
        if commit_hash is not None and local_dir is None:
            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if os.path.exists(snapshot_folder):
                # The folder exists, but may be partial (e.g. after an interrupted download): only return it
                # if the cached tree listing confirms it is complete.
                _raise_if_incomplete_snapshot(
                    tree_cache_folder=tree_cache_folder,
                    commit_hash=commit_hash,
                    base_dir=snapshot_folder,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    repo_id=repo_id,
                    revision=revision,
                    api_call_error=api_call_error,
                )
                return snapshot_folder

        # If local_dir is not None, return it if it exists and is complete
        if local_dir is not None:
            local_dir = Path(local_dir)
            if local_dir.is_dir() and any(local_dir.iterdir()):
                if commit_hash is not None:
                    _raise_if_incomplete_snapshot(
                        tree_cache_folder=tree_cache_folder,
                        commit_hash=commit_hash,
                        base_dir=str(local_dir),
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        repo_id=repo_id,
                        revision=revision,
                        api_call_error=api_call_error,
                    )
                logger.warning(
                    f"Returning existing local_dir `{local_dir}` as remote repo cannot be accessed in `snapshot_download` ({api_call_error})."
                )
                return str(local_dir.resolve())
        # If we couldn't find the appropriate folder on disk, raise an error.
        if local_files_only:
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass "
                "'local_files_only=False' as input."
            )
        elif isinstance(api_call_error, OfflineModeIsEnabled):
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, set "
                "'HF_HUB_OFFLINE=0' as environment variable."
            ) from api_call_error
        elif isinstance(api_call_error, (RepositoryNotFoundError, GatedRepoError)) or (
            isinstance(api_call_error, HfHubHTTPError) and api_call_error.response.status_code == 401
        ):
            # Repo not found, gated, or specific authentication error => let's raise the actual error
            raise api_call_error
        else:
            # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
            raise LocalEntryNotFoundError(
                f"Got: {api_call_error.__class__.__name__}: {api_call_error}"
                "\nAn error happened while trying to locate the files on the Hub, and we cannot find the appropriate"
                " snapshot folder for the specified revision on the local disk. Please check your internet connection"
                " and try again."
            ) from api_call_error

    # At this stage, the commit hash is known and internet connection is up and running
    # => let's download the files!
    assert commit_hash is not None

    filtered_repo_files = list(
        filter_repo_objects(
            items=tree_entries.keys(),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    if tree_entries and not filtered_repo_files and (allow_patterns is not None or ignore_patterns is not None):
        provided_patterns = ", ".join(
            f"{name}={value!r}"
            for name, value in (("allow_patterns", allow_patterns), ("ignore_patterns", ignore_patterns))
            if value is not None
        )
        logger.warning(f"No files matched {provided_patterns}.")
    tqdm_desc = f"Fetching {len(filtered_repo_files)} files"
    if dry_run:
        tqdm_desc = "[dry-run] " + tqdm_desc

    snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
    # if passed revision is not identical to commit_hash
    # then revision has to be a branch name or tag name.
    # In that case store a ref (except if ResolvedRevision, in which case it's already done).
    if not isinstance(revision, ResolvedRevision):
        try:
            # Skips the write if the ref is already up to date, and writes atomically otherwise so that
            # concurrent readers never observe a truncated (empty) ref file.
            _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)
        except OSError as e:
            logger.warning(
                f"Ignored error while writing commit hash to {os.path.join(storage_folder, 'refs', revision)}: {e}."
            )

    results: list[str | DryRunFileInfo] = []

    # User can use its own tqdm class or the default one from `huggingface_hub.utils`
    tqdm_class = tqdm_class or hf_tqdm

    # Create progress bars for the bytes downloaded.
    # Transfer bytes are received from the network; reconstruction bytes are written to disk.
    transfer_progress = _create_progress_bar(
        cls=tqdm_class,
        log_level=logger.getEffectiveLevel(),
        name="huggingface_hub.snapshot_download.transfer",
        desc="Downloading bytes",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        bar_format=XET_TRANSFER_BAR_FORMAT,
    )

    reconstruct_progress = _create_progress_bar(
        cls=tqdm_class,
        log_level=logger.getEffectiveLevel(),
        name="huggingface_hub.snapshot_download",
        desc="Reconstructing (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        bar_format=XET_BYTES_BAR_FORMAT,
    )

    class _AggregatedTqdm:
        """Fake tqdm object to aggregate progress into the parent snapshot progress bars.

        In practice, the `_AggregatedTqdm` object won't be displayed; it's just used to update
        the `reconstruct_progress` and `transfer_progress` bars from each thread/file download.
        """

        def __init__(self, *args, **kwargs):
            # Adjust the total of the parent progress bar
            total = kwargs.pop("total", None)
            if total is not None:
                reconstruct_progress.total = (reconstruct_progress.total or 0) + total
                transfer_progress.total = (transfer_progress.total or 0) + total
                reconstruct_progress.refresh()

            # Adjust initial of the parent progress bar
            initial = kwargs.pop("initial", 0)
            if initial:
                reconstruct_progress.update(initial)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def close(self) -> None:
            pass

        def update(self, n: int | float | None = 1) -> None:
            reconstruct_progress.update(n)

        def update_transfer(self, n: int | float | None = 1) -> None:
            _update_transfer_bar(transfer_progress, int(n or 0))

        def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
            # Discard the caller's per-file rate; report the summed rate across all files instead.
            _set_aggregate_rate_postfix(reconstruct_progress)

        def set_transfer_postfix_str(self, postfix: str, refresh: bool = False) -> None:
            _set_aggregate_rate_postfix(transfer_progress)

    # Pass the commit_hash as revision to hf_hub_download to skip network call if:
    # - file is cached
    # - or xet file with metadata cached in /tree cache
    def _inner_hf_hub_download(repo_file: str) -> None:
        results.append(
            hf_hub_download(  # type: ignore
                repo_id,
                filename=repo_file,
                repo_type=repo_type,
                revision=commit_hash,
                endpoint=endpoint,
                cache_dir=cache_dir,
                local_dir=local_dir,
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
                etag_timeout=etag_timeout,
                force_download=force_download,
                token=token,
                headers=headers,
                tqdm_class=_AggregatedTqdm,  # type: ignore
                dry_run=dry_run,
            )
        )

    hf_thread_map(
        _inner_hf_hub_download,
        filtered_repo_files,
        desc=tqdm_desc,
        max_workers=max_workers,
        tqdm_class=tqdm_class,
    )

    _finish_transfer_bar(transfer_progress)
    transfer_progress.set_description_str("Download complete")
    reconstruct_progress.set_description("Reconstruction complete")

    if dry_run:
        assert all(isinstance(r, DryRunFileInfo) for r in results)
        return results  # type: ignore

    if local_dir is not None:
        return str(os.path.realpath(local_dir))
    return snapshot_folder


def _raise_if_incomplete_snapshot(
    *,
    tree_cache_folder: str,
    commit_hash: str,
    base_dir: str,
    allow_patterns: list[str] | str | None,
    ignore_patterns: list[str] | str | None,
    repo_id: str,
    revision: str,
    api_call_error: Exception | None,
) -> None:
    """Raise [`IncompleteSnapshotError`] if the cached tree listing shows `base_dir` misses requested files.

    If the tree listing is not cached we cannot tell, so we do nothing and the caller keeps returning the
    folder as-is. Otherwise every expected file (after pattern filtering) must exist under `base_dir`.
    """
    tree_entries = read_tree_cache(tree_cache_folder, commit_hash)
    if tree_entries is None:
        return
    expected = filter_repo_objects(
        items=tree_entries.keys(), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
    )
    missing = [path for path in expected if not _local_file_exists(base_dir, path)]
    if not missing:
        return

    sample = ", ".join(missing[:3])
    if len(missing) > 3:
        sample += f", ... ({len(missing) - 3} more)"
    if api_call_error is not None:
        reason = f"The Hub could not be reached ({api_call_error.__class__.__name__}: {api_call_error})."
    else:
        reason = "Outgoing traffic is disabled ('local_files_only=True')."
    raise IncompleteSnapshotError(
        f"The cached snapshot for '{repo_id}' (revision '{revision}', commit {commit_hash}) is incomplete: "
        f"{len(missing)} file(s) are missing ({sample}). {reason} Re-run the download with network access "
        "to complete the snapshot.",
        snapshot_path=base_dir,
    ) from api_call_error


@validate_hf_hub_args
def get_cached_repo_tree(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> list[RepoFile]:
    """Return the cached tree listing of a repo at a given revision, without any network call.

    The tree listing is the set of files (with their download metadata) of a repo at a commit. It is populated
    on disk as a side effect of [`snapshot_download`] (see the `trees/<commit_hash>.json` cache files) and is
    used to skip network calls on subsequent downloads. This function exposes that cache directly.

    If you need the current tree listing of a repo on the Hub, use [`list_repo_tree`] instead.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        repo_type (`str`, *optional*):
            Set to `"dataset"`, `"space"` or `"kernel"` if listing from a dataset, space or kernel repo,
            `None` or `"model"` if listing from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id, which can be a branch name, a tag, or a commit hash. Defaults to the
            default branch. Branch/tag names are resolved to a commit hash using the local cache (`refs/`).
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored. Defaults to the value of `HF_HUB_CACHE`.
        local_dir (`str` or `Path`, *optional*):
            If provided, read the tree listing cached by a `local_dir` download (from
            `local_dir/.cache/huggingface/`) instead of the main cache. Branch/tag revisions are still resolved
            to a commit hash using the main cache (`cache_dir`).

    Returns:
        `list[RepoFile]`: The list of [`RepoFile`] objects cached for this revision.

    Raises:
        [`~errors.CachedRepoTreeNotFoundError`]
            If no tree listing is cached for the requested revision (e.g. the repo was never downloaded at this revision).

    Example:
        ```py
        >>> from huggingface_hub import get_cached_repo_tree
        >>> files = get_cached_repo_tree("openai-community/gpt2")
        >>> [f.path for f in files]
        ['.gitattributes', 'config.json', 'model.safetensors', ...]
        ```
    """
    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())
    if local_dir is not None:
        local_dir = str(Path(local_dir).expanduser().resolve())
    if revision is None:
        revision = constants.DEFAULT_REVISION
    if repo_type is None:
        repo_type = constants.REPO_TYPE_MODEL
    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES_WITH_KERNEL)}"
        )

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    # For `local_dir` downloads the tree listing lives under `local_dir/.cache/huggingface/`; otherwise it lives
    # in the per-repo `storage_folder`. Refs are always recorded in the main cache, so we resolve them there.
    tree_cache_folder = tree_cache_folder_for_local_dir(local_dir) if local_dir is not None else storage_folder

    # The tree cache is keyed by commit hash. Resolve the revision to a commit hash: either it already is one,
    # or it's a branch/tag name recorded in `refs/` by a previous download.
    if isinstance(revision, ResolvedRevision):
        commit_hash = revision.resolved
    elif REGEX_COMMIT_HASH.fullmatch(revision):
        commit_hash = revision
    else:
        ref_path = os.path.join(storage_folder, "refs", revision)
        if not os.path.isfile(ref_path):
            raise CachedRepoTreeNotFoundError(
                f"No cached tree listing found for '{repo_id}' (revision '{revision}', repo_type '{repo_type}'): "
                f"the revision is not a commit hash and no matching ref is cached in '{storage_folder}'. "
                "Download the repo (e.g. with `snapshot_download`) to populate the cache first."
            )
        with open(ref_path) as f:
            commit_hash = f.read()

    tree_entries = read_tree_cache(tree_cache_folder, commit_hash)
    if tree_entries is None:
        raise CachedRepoTreeNotFoundError(
            f"No cached tree listing found for '{repo_id}' (revision '{revision}', commit '{commit_hash}', "
            f"repo_type '{repo_type}') in '{tree_cache_folder}'. Download the repo (e.g. with `snapshot_download`) "
            "to populate the cache first."
        )

    return [
        RepoFile(path=path, size=entry.size, oid=entry.blob_id, xetHash=entry.xet_hash)
        for path, entry in tree_entries.items()
    ]


def _local_file_exists(base_dir: str, path: str) -> bool:
    """Check whether a repo file (path relative to `base_dir`, '/'-separated) exists on disk.

    On Windows, paths longer than 255 characters must be prefixed with `\\\\?\\`, otherwise `os.path.isfile` reports an
    existing file as missing.
    """
    return os.path.isfile(as_extended_path(os.path.join(base_dir, *path.split("/"))))
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
"""Contains `WebhooksServer` and `webhook_endpoint` to create a webhook server easily."""

import atexit
import inspect
import os
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

from .utils import experimental, is_fastapi_available, is_gradio_available


if TYPE_CHECKING:
    import gradio as gr
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse


_global_app: Optional["WebhooksServer"] = None  # ty: ignore[invalid-type-form]
_is_local = os.environ.get("SPACE_ID") is None


@experimental
class WebhooksServer:
    """
    The [`WebhooksServer`] class lets you create an instance of a Gradio app that can receive Huggingface webhooks.
    These webhooks can be registered using the [`~WebhooksServer.add_webhook`] decorator. Webhook endpoints are added to
    the app as a POST endpoint to the FastAPI router. Once all the webhooks are registered, the `launch` method has to be
    called to start the app.

    It is recommended to accept [`WebhookPayload`] as the first argument of the webhook function. It is a Pydantic
    model that contains all the information about the webhook event. The data will be parsed automatically for you.

    Check out the [webhooks guide](../guides/webhooks_server) for a step-by-step tutorial on how to set up your
    WebhooksServer and deploy it on a Space.

    > [!WARNING]
    > `WebhooksServer` is experimental. Its API is subject to change in the future.

    > [!WARNING]
    > You must have `gradio` installed to use `WebhooksServer` (`pip install --upgrade gradio`).

    Args:
        ui (`gradio.Blocks`, optional):
            A Gradio UI instance to be used as the Space landing page. If `None`, a UI displaying instructions
            about the configured webhooks is created.
        webhook_secret (`str`, optional):
            A secret key to verify incoming webhook requests. You can set this value to any secret you want as long as
            you also configure it in your [webhooks settings panel](https://huggingface.co/settings/webhooks). You
            can also set this value as the `WEBHOOK_SECRET` environment variable. If no secret is provided, the
            webhook endpoints are opened without any security.

    Example:

        ```python
        import gradio as gr
        from huggingface_hub import WebhooksServer, WebhookPayload

        with gr.Blocks() as ui:
            ...

        app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

        @app.add_webhook("/say_hello")
        async def hello(payload: WebhookPayload):
            return {"message": "hello"}

        app.launch()
        ```
    """

    def __new__(cls, *args, **kwargs) -> "WebhooksServer":  # ty: ignore[invalid-type-form]
        if not is_gradio_available():
            raise ImportError(
                "You must have `gradio` installed to use `WebhooksServer`. Please run `pip install --upgrade gradio`"
                " first."
            )
        if not is_fastapi_available():
            raise ImportError(
                "You must have `fastapi` installed to use `WebhooksServer`. Please run `pip install --upgrade fastapi`"
                " first."
            )
        _load_fastapi()
        return super().__new__(cls)

    def __init__(
        self,
        ui: Optional["gr.Blocks"] = None,
        webhook_secret: str | None = None,
    ) -> None:
        self._ui = ui

        self.webhook_secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
        self.registered_webhooks: dict[str, Callable] = {}
        _warn_on_empty_secret(self.webhook_secret)

    def add_webhook(self, path: str | None = None) -> Callable:
        """
        Decorator to add a webhook to the [`WebhooksServer`] server.

        Args:
            path (`str`, optional):
                The URL path to register the webhook function. If not provided, the function name will be used as the
                path. In any case, all webhooks are registered under `/webhooks`.

        Raises:
            ValueError: If the provided path is already registered as a webhook.

        Example:
            ```python
            from huggingface_hub import WebhooksServer, WebhookPayload

            app = WebhooksServer()

            @app.add_webhook
            async def trigger_training(payload: WebhookPayload):
                if payload.repo.type == "dataset" and payload.event.action == "update":
                    # Trigger a training job if a dataset is updated
                    ...

            app.launch()
        ```
        """
        # Usage: directly as decorator. Example: `@app.add_webhook`
        if callable(path):
            # If path is a function, it means it was used as a decorator without arguments
            return self.add_webhook()(path)

        # Usage: provide a path. Example: `@app.add_webhook(...)`
        @wraps(FastAPI.post)
        def _inner_post(*args, **kwargs):
            func = args[0]
            abs_path = f"/webhooks/{(path or func.__name__).strip('/')}"
            if abs_path in self.registered_webhooks:
                raise ValueError(f"Webhook {abs_path} already exists.")
            self.registered_webhooks[abs_path] = func

        return _inner_post

    def launch(self, prevent_thread_lock: bool = False, **launch_kwargs: Any) -> None:
        """Launch the Gradio app and register webhooks to the underlying FastAPI server.

        Input parameters are forwarded to Gradio when launching the app.
        """
        ui = self._ui or self._get_default_ui()

        # Start Gradio App
        #   - as non-blocking so that webhooks can be added afterwards
        #   - as shared if launch locally (to debug webhooks)
        launch_kwargs.setdefault("share", _is_local)
        self.fastapi_app, _, _ = ui.launch(prevent_thread_lock=True, **launch_kwargs)

        # Register webhooks to FastAPI app
        for path, func in self.registered_webhooks.items():
            # Add secret check if required
            if self.webhook_secret is not None:
                func = _wrap_webhook_to_check_secret(func, webhook_secret=self.webhook_secret)

            # Add route to FastAPI app
            self.fastapi_app.post(path)(func)

        # Print instructions and block main thread
        space_host = os.environ.get("SPACE_HOST")
        url = "https://" + space_host if space_host is not None else (ui.share_url or ui.local_url)
        if url is None:
            raise ValueError("Cannot find the URL of the app. Please provide a valid `ui` or update `gradio` version.")
        url = url.strip("/")
        message = "\nWebhooks are correctly setup and ready to use:"
        message += "\n" + "\n".join(f"  - POST {url}{webhook}" for webhook in self.registered_webhooks)
        message += "\nGo to https://huggingface.co/settings/webhooks to setup your webhooks."
        print(message)

        if not prevent_thread_lock:
            ui.block_thread()

    def _get_default_ui(self) -> "gr.Blocks":
        """Default UI if not provided (lists webhooks and provides basic instructions)."""
        import gradio as gr

        with gr.Blocks() as ui:
            gr.Markdown("# This is an app to process 🤗 Webhooks")
            gr.Markdown(
                "Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on"
                " specific repos or to all repos belonging to particular set of users/organizations (not just your"
                " repos, but any repo). Check out this [guide](https://huggingface.co/docs/hub/webhooks) to get to"
                " know more about webhooks on the Huggingface Hub."
            )
            gr.Markdown(
                f"{len(self.registered_webhooks)} webhook(s) are registered:"
                + "\n\n"
                + "\n ".join(
                    f"- [{webhook_path}]({_get_webhook_doc_url(webhook.__name__, webhook_path)})"
                    for webhook_path, webhook in self.registered_webhooks.items()
                )
            )
            gr.Markdown(
                "Go to https://huggingface.co/settings/webhooks to setup your webhooks."
                + "\nYou app is running locally. Please look at the logs to check the full URL you need to set."
                if _is_local
                else (
                    "\nThis app is running on a Space. You can find the corresponding URL in the options menu"
                    " (top-right) > 'Embed the Space'. The URL looks like 'https://{username}-{repo_name}.hf.space'."
                )
            )
        return ui


@experimental
def webhook_endpoint(path: str | None = None) -> Callable:
    """Decorator to start a [`WebhooksServer`] and register the decorated function as a webhook endpoint.

    This is a helper to get started quickly. If you need more flexibility (custom landing page or webhook secret),
    you can use [`WebhooksServer`] directly. You can register multiple webhook endpoints (to the same server) by using
    this decorator multiple times.

    Check out the [webhooks guide](../guides/webhooks_server) for a step-by-step tutorial on how to set up your
    server and deploy it on a Space.

    > [!WARNING]
    > `webhook_endpoint` is experimental. Its API is subject to change in the future.

    > [!WARNING]
    > You must have `gradio` installed to use `webhook_endpoint` (`pip install --upgrade gradio`).

    Args:
        path (`str`, optional):
            The URL path to register the webhook function. If not provided, the function name will be used as the path.
            In any case, all webhooks are registered under `/webhooks`.

    Examples:
        The default usage is to register a function as a webhook endpoint. The function name will be used as the path.
        The server will be started automatically at exit (i.e. at the end of the script).

        ```python
        from huggingface_hub import webhook_endpoint, WebhookPayload

        @webhook_endpoint
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...

        # Server is automatically started at the end of the script.
        ```

        Advanced usage: register a function as a webhook endpoint and start the server manually. This is useful if you
        are running it in a notebook.

        ```python
        from huggingface_hub import webhook_endpoint, WebhookPayload

        @webhook_endpoint
        async def trigger_training(payload: WebhookPayload):
            if payload.repo.type == "dataset" and payload.event.action == "update":
                # Trigger a training job if a dataset is updated
                ...

        # Start the server manually
        trigger_training.launch()
        ```
    """
    if callable(path):
        # If path is a function, it means it was used as a decorator without arguments
        return webhook_endpoint()(path)

    @wraps(WebhooksServer.add_webhook)
    def _inner(func: Callable) -> Callable:
        app = _get_global_app()
        app.add_webhook(path)(func)
        if len(app.registered_webhooks) == 1:
            # Register `app.launch` to run at exit (only once)
            atexit.register(app.launch)

        @wraps(app.launch)
        def _launch_now():
            # Run the app directly (without waiting atexit)
            atexit.unregister(app.launch)
            app.launch()

        func.launch = _launch_now  # type: ignore
        return func

    return _inner


def _get_global_app() -> WebhooksServer:  # ty: ignore[invalid-type-form]
    global _global_app
    if _global_app is None:
        _global_app = WebhooksServer()
    return _global_app


def _load_fastapi() -> None:
    global FastAPI, JSONResponse, Request

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse


def _warn_on_empty_secret(webhook_secret: str | None) -> None:
    if webhook_secret is None:
        print("Webhook secret is not defined. This means your webhook endpoints will be open to everyone.")
        print(
            "To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization: "
            "\n\t`app = WebhooksServer(webhook_secret='my_secret', ...)`"
        )
        print(
            "For more details about webhook secrets, please refer to"
            " https://huggingface.co/docs/hub/webhooks#webhook-secret."
        )
    else:
        print("Webhook secret is correctly defined.")


def _get_webhook_doc_url(webhook_name: str, webhook_path: str) -> str:
    """Returns the anchor to a given webhook in the docs (experimental)"""
    return "/docs#/default/" + webhook_name + webhook_path.replace("/", "_") + "_post"


def _wrap_webhook_to_check_secret(func: Callable, webhook_secret: str) -> Callable:
    """Wraps a webhook function to check the webhook secret before calling the function.

    This is a hacky way to add the `request` parameter to the function signature. Since FastAPI based itself on route
    parameters to inject the values to the function, we need to hack the function signature to retrieve the `Request`
    object (and hence the headers). A far cleaner solution would be to use a middleware. However, since
    `fastapi==0.90.1`, a middleware cannot be added once the app has started. And since the FastAPI app is started by
    Gradio internals (and not by us), we cannot add a middleware.

    This method is called only when a secret has been defined by the user. If a request is sent without the
    "x-webhook-secret", the function will return a 401 error (unauthorized). If the header is sent but is incorrect,
    the function will return a 403 error (forbidden).

    Inspired by https://stackoverflow.com/a/33112180.
    """
    initial_sig = inspect.signature(func)

    @wraps(func)
    async def _protected_func(request: Request, **kwargs):
        request_secret = request.headers.get("x-webhook-secret")
        if request_secret is None:
            return JSONResponse({"error": "x-webhook-secret header not set."}, status_code=401)
        if request_secret != webhook_secret:
            return JSONResponse({"error": "Invalid webhook secret."}, status_code=403)

        # Inject `request` in kwargs if required
        if "request" in initial_sig.parameters:
            kwargs["request"] = request

        # Handle both sync and async routes
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)

    # Update signature to include request
    if "request" not in initial_sig.parameters:
        _protected_func.__signature__ = initial_sig.replace(  # type: ignore
            parameters=(
                inspect.Parameter(name="request", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
            )
            + tuple(initial_sig.parameters.values())
        )

    # Return protected route
    return _protected_func
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
"""Contains data structures to parse the webhooks payload."""

from typing import Literal

from .utils import is_pydantic_available


if is_pydantic_available():
    from pydantic import BaseModel
else:
    # Define a dummy BaseModel to avoid import errors when pydantic is not installed
    # Import error will be raised when trying to use the class

    class BaseModel:  # type: ignore [no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "You must have `pydantic` installed to use `WebhookPayload`. This is an optional dependency that"
                " should be installed separately. Please run `pip install --upgrade pydantic` and retry."
            )


# This is an adaptation of the ReportV3 interface implemented in moon-landing. V0, V1 and V2 have been ignored as they
# are not in used anymore. To keep in sync when format is updated in
# https://github.com/huggingface/moon-landing/blob/main/server/lib/HFWebhooks.ts (internal link).


WebhookEvent_T = Literal[
    "create",
    "delete",
    "move",
    "update",
]
RepoChangeEvent_T = Literal[
    "add",
    "move",
    "remove",
    "update",
]
RepoType_T = Literal[
    "dataset",
    "model",
    "space",
]
DiscussionStatus_T = Literal[
    "closed",
    "draft",
    "open",
    "merged",
]
SupportedWebhookVersion = Literal[3]


class ObjectId(BaseModel):
    id: str


class WebhookPayloadUrl(BaseModel):
    web: str
    api: str | None = None


class WebhookPayloadMovedTo(BaseModel):
    name: str
    owner: ObjectId


class WebhookPayloadWebhook(ObjectId):
    version: SupportedWebhookVersion


class WebhookPayloadEvent(BaseModel):
    action: WebhookEvent_T
    scope: str


class WebhookPayloadDiscussionChanges(BaseModel):
    base: str
    mergeCommitId: str | None = None


class WebhookPayloadComment(ObjectId):
    author: ObjectId
    hidden: bool
    content: str | None = None
    url: WebhookPayloadUrl


class WebhookPayloadDiscussion(ObjectId):
    num: int
    author: ObjectId
    url: WebhookPayloadUrl
    title: str
    isPullRequest: bool
    status: DiscussionStatus_T
    changes: WebhookPayloadDiscussionChanges | None = None
    pinned: bool | None = None


class WebhookPayloadRepo(ObjectId):
    owner: ObjectId
    head_sha: str | None = None
    name: str
    private: bool
    subdomain: str | None = None
    tags: list[str] | None = None
    type: Literal["dataset", "model", "space"]
    url: WebhookPayloadUrl


class WebhookPayloadUpdatedRef(BaseModel):
    ref: str
    oldSha: str | None = None
    newSha: str | None = None


class WebhookPayload(BaseModel):
    event: WebhookPayloadEvent
    repo: WebhookPayloadRepo
    discussion: WebhookPayloadDiscussion | None = None
    comment: WebhookPayloadComment | None = None
    webhook: WebhookPayloadWebhook
    movedTo: WebhookPayloadMovedTo | None = None
    updatedRefs: list[WebhookPayloadUpdatedRef] | None = None
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
"""Best-effort local cache for SandboxPool hosts (host/pool mode)."""

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

from . import constants
from .utils import WeakFileLock, logging


logger = logging.get_logger(__name__)

# Bump if the on-disk layout changes incompatibly; older/newer files are ignored on read.
_CACHE_VERSION = 2

# A write should never block a sandbox creation for long: the cache is best-effort, so we
# rather skip persisting than wait on a stuck lock.
_LOCK_TIMEOUT = 5.0

# How long a cached host may be turned into a credentialed transport on the strength of the
# file alone. Within it, the `pool create` -> `create --pool` sequence stays at zero extra
# round-trips (the point of the cache); past it, the caller re-checks the host against the
# Jobs API before sending it anything.
HOST_TRUST_TTL = 15 * 60

# The cache holds reusable host URLs and the public nonces the host tokens derive from, so
# both the directories and the files are private to the user.
_DIR_MODE = 0o700

# Domain separation, so a digest from this cache is not also a digest of the same material
# computed for some other purpose (and so a version bump renames every directory).
_DIGEST_PREFIX = f"hf-sandbox-pool-cache/v{_CACHE_VERSION}"

# Plausibility bounds for the schema check below. Generous on purpose: this is here to turn a
# wrong-typed or absurd value into a cache miss, not to re-specify the pool's own limits.
_MAX_COUNT = 1_000_000
_MAX_SECONDS = 10 * 365 * 24 * 3600
_MAX_TIMESTAMP = 4_000_000_000  # epoch seconds, some time in 2096
_MAX_STR = 2048
# Job ids are backend-assigned; the id also feeds a URL and an `inspect_job` call, so keep it
# to characters that cannot change the meaning of either.
_JOB_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
# Nonces are minted by `token_hex(16)`; the bounds are loose so that changing that length
# would not quietly turn every cache entry into a miss.
_NONCE_RE = re.compile(r"[0-9a-fA-F]{16,128}")


def _digest(*parts: str) -> str:
    """Short, stable digest of `parts` (which are joined unambiguously)."""
    material = "\x00".join((_DIGEST_PREFIX, *parts))
    return hashlib.sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class CacheContext:
    """The security context a cache entry belongs to.

    A cached entry is not just data: the fast path rebuilds a host transport from it and
    sends it the HF bearer plus the derived host token. So it may only ever be read back by
    the same *endpoint*, *credential* and *namespace* that wrote it -- anything else is a
    plain cache miss, which is what the docs promise and what keeps a `connect(namespace=...)`
    from being served hosts that belong to another namespace.

    `principal` is an opaque, stable id for the credential; the token itself is never stored
    nor used as a key (see `for_credential`).
    """

    endpoint: str
    principal: str
    namespace: str | None

    @classmethod
    def for_credential(cls, *, endpoint: str, token: str, namespace: str | None) -> "CacheContext":
        """Build the context of a given HF token, identified by its fingerprint.

        The fingerprint is a local computation, which is what keeps a cache hit at zero
        round-trips -- resolving the credential to a user id would cost a `whoami` and defeat
        the purpose of the cache. It is derived from the token but does not contain it: only
        the digest is ever written to disk.

        A rotated or swapped credential therefore reads as a different principal and gets a
        cache miss (the cold path, which always works), rather than someone else's hosts.
        """
        return cls(endpoint=endpoint, principal=_digest("principal", token)[:32], namespace=namespace)

    @property
    def key(self) -> str:
        """Digest of the whole context; the name of the cache subdirectory it owns."""
        return _digest("context", self.endpoint, self.principal, self.namespace or "")[:16]


@dataclass
class CachedHost:
    """A single host Job of a pool, as last seen by some process.

    `base_url` + `nonce` are everything needed to rebuild the in-job server transport
    (`_SandboxServer`) without an `inspect_job` round-trip: the per-sandbox auth token is
    re-derived from the user's HF token and `nonce` (see `_derive_sandbox_token`).
    """

    job_id: str
    owner: str  # namespace the host job runs under (for cancel/inspect)
    base_url: str  # exposed sbx-server URL (does not change while the job lives)
    nonce: str  # public nonce from the job label; derives the sandbox auth token
    capacity: int  # SBX_CAPACITY: max sandboxes the host packs
    live: int  # sandboxes last observed on the host (best-effort, may be stale)
    updated_at: float = 0.0


@dataclass
class PoolCache:
    """Cached view of one pool: its config (to boot new hosts) + its known hosts."""

    pool_id: str
    image: str
    flavor: str
    sandboxes_per_host: int
    max_hosts: int | None
    idle_timeout: int | None
    namespace: str | None
    # Context the entry was written in, so a file that is moved (or copied from elsewhere)
    # into a directory it was not written for is still rejected. See `CacheContext`.
    endpoint: str = ""
    context: str = ""
    hosts: List[CachedHost] = field(default_factory=list)
    version: int = _CACHE_VERSION
    updated_at: float = 0.0


def _pools_dir() -> Path:
    return Path(constants.HF_HOME) / "sandbox" / "pools"


def _check_pool_id(pool_id: str) -> str:
    """Return `pool_id` if it is safe to use as a file name, else raise."""
    if any(c in pool_id for c in ("/", "\\", "\x00")) or pool_id in (".", ".."):
        raise ValueError(f"Invalid pool id: {pool_id!r}")
    return pool_id


def pool_cache_path(pool_id: str, context: CacheContext) -> Path:
    """Path of the cache file for `pool_id` in `context` (no I/O)."""
    return _pools_dir() / context.key / f"{_check_pool_id(pool_id)}.json"


def read_pool_cache(pool_id: str, context: CacheContext) -> PoolCache | None:
    """Return the cached view of `pool_id` for `context`, or `None`.

    Missing, corrupt, incompatible, mistyped and foreign-context files are all the same thing
    here: a cache miss, logged at debug level. The caller's cold path is always correct, so
    this never raises -- a bad cache costs latency, never an error mid-`create()`.
    """
    try:
        path = pool_cache_path(pool_id, context)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            return None
        hosts = [CachedHost(**h) for h in data.pop("hosts", [])]
        cache = PoolCache(**data, hosts=hosts)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Ignoring unreadable sandbox pool cache for {pool_id!r}: {e}")
        return None
    reason = _cache_rejection(cache, pool_id=pool_id, context=context)
    if reason is not None:
        logger.debug(f"Ignoring sandbox pool cache for {pool_id!r}: {reason}.")
        return None
    return cache


def save_pool_cache(
    pool_id: str,
    *,
    context: CacheContext,
    image: str,
    flavor: str,
    sandboxes_per_host: int,
    max_hosts: int | None,
    idle_timeout: int | None,
    hosts: List[CachedHost],
    dead_host_ids: set[str] | None = None,
) -> None:
    """Merge `hosts` into the cache for `pool_id` in `context` (best-effort, never raises).

    Concurrency-safe: under a file lock, the on-disk hosts are read, then `hosts` are
    upserted by `job_id` and `dead_host_ids` removed, so a process only adds/updates what
    it learned and never drops hosts another process discovered. The result is written
    atomically. The pool config is refreshed from the arguments.
    """
    dead = dead_host_ids or set()
    try:
        path = pool_cache_path(pool_id, context)
        _ensure_private_dir(path.parent)
        with WeakFileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT):
            existing = read_pool_cache(pool_id, context)
            merged = {h.job_id: h for h in (existing.hosts if existing else [])}
            for host in hosts:
                merged[host.job_id] = host
            for job_id in dead:
                merged.pop(job_id, None)
            cache = PoolCache(
                pool_id=pool_id,
                image=image,
                flavor=flavor,
                sandboxes_per_host=sandboxes_per_host,
                max_hosts=max_hosts,
                idle_timeout=idle_timeout,
                namespace=context.namespace,
                endpoint=context.endpoint,
                context=context.key,
                hosts=list(merged.values()),
                updated_at=time.time(),
            )
            _atomic_write(path, cache)
    except Exception as e:
        logger.debug(f"Could not write sandbox pool cache for {pool_id!r}: {e}")


def delete_pool_cache(pool_id: str, context: CacheContext | None = None) -> None:
    """Remove the cache file(s) for `pool_id` (best-effort, never raises).

    Without a `context`, the pool's entry is removed from *every* context: deleting a pool is
    the one operation for which over-deleting is the safe direction (the cache is disposable),
    and the caller may not know which credential wrote the entry.
    """
    try:
        if context is not None:
            pool_cache_path(pool_id, context).unlink(missing_ok=True)
            return
        filename = f"{_check_pool_id(pool_id)}.json"
        for directory in _pools_dir().iterdir():
            if directory.is_dir():
                (directory / filename).unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Could not delete sandbox pool cache for {pool_id}: {e}")


def _ensure_private_dir(path: Path) -> None:
    """Create the cache directory tree, readable by this user only.

    `mkdir`'s mode is masked by the umask and the tree may predate this code, so the modes
    are also set explicitly, from the leaf up to (and including) the sandbox directory.
    """
    path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    root = Path(constants.HF_HOME) / "sandbox"
    for directory in (path, *path.parents):
        if directory != root and root not in directory.parents:
            break  # $HF_HOME itself (and anything above it) is not ours to re-mode
        os.chmod(directory, _DIR_MODE)


def _atomic_write(path: Path, cache: PoolCache) -> None:
    """Write the cache via a temp file + `os.replace` so readers never see a partial file.

    `mkstemp` gives the temp file an unpredictable name and mode `0600` regardless of the
    umask (which `os.replace` then carries over to the final file), so no other user can read
    -- or pre-create and win a race on -- the host URLs and nonces on their way to disk.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(cache), f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_count(value: Any, *, minimum: int = 0, maximum: int = _MAX_COUNT) -> bool:
    """Whether `value` is a plausible count. `bool` is an `int`, but not a count."""
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _is_timestamp(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= _MAX_TIMESTAMP


def _is_text(value: Any, *, allow_empty: bool = False) -> bool:
    return isinstance(value, str) and (allow_empty or len(value) > 0) and len(value) <= _MAX_STR


def _cache_rejection(cache: PoolCache, *, pool_id: str, context: CacheContext) -> str | None:
    """Why this cache file must not be used, or `None` if it may be.

    Two jobs. First, the entry has to belong to the caller's security context -- same
    endpoint, same credential, same namespace -- which is what makes a cross-namespace or
    cross-principal hit a miss instead of a confidentiality bug.

    Second, a schema check. `dataclass` does not enforce types, so a well-shaped file with
    `"capacity": "50"` parses happily and only blows up much later, on `host.capacity -
    host.live`, as an uncaught `TypeError` in the middle of `create()`. Validating here turns
    that into the documented cache miss.
    """
    if cache.context != context.key:
        return "it was written for a different endpoint, credential or namespace"
    if cache.endpoint != context.endpoint:
        return f"it names endpoint {cache.endpoint!r}, not {context.endpoint!r}"
    if cache.namespace != context.namespace:
        return f"it names namespace {cache.namespace!r}, not {context.namespace!r}"
    if cache.pool_id != pool_id:
        return f"it names pool {cache.pool_id!r}, not {pool_id!r}"

    if not _is_text(cache.image) or not _is_text(cache.flavor):
        return "its image/flavor are not strings"
    if not _is_count(cache.sandboxes_per_host, minimum=1):
        return f"its sandboxes_per_host is not a positive count: {cache.sandboxes_per_host!r}"
    if cache.max_hosts is not None and not _is_count(cache.max_hosts, minimum=1):
        return f"its max_hosts is not a positive count: {cache.max_hosts!r}"
    if cache.idle_timeout is not None and not _is_count(cache.idle_timeout, maximum=_MAX_SECONDS):
        return f"its idle_timeout is not a duration in seconds: {cache.idle_timeout!r}"
    if not _is_timestamp(cache.updated_at):
        return f"its updated_at is not a timestamp: {cache.updated_at!r}"

    if not isinstance(cache.hosts, list):
        return "its hosts are not a list"
    for host in cache.hosts:
        if not _JOB_ID_RE.fullmatch(host.job_id if isinstance(host.job_id, str) else ""):
            return f"host {host.job_id!r} is not a plausible job id"
        if not _JOB_ID_RE.fullmatch(host.owner if isinstance(host.owner, str) else ""):
            return f"host {host.job_id} names a non-namespace owner: {host.owner!r}"
        if not _is_text(host.base_url):
            return f"host {host.job_id} has no URL"
        if not isinstance(host.nonce, str) or not _NONCE_RE.fullmatch(host.nonce):
            return f"host {host.job_id} has no usable nonce"
        if not _is_count(host.capacity) or not _is_count(host.live):
            return f"host {host.job_id} has a non-numeric capacity/live count"
        if not _is_timestamp(host.updated_at):
            return f"host {host.job_id} has no usable timestamp"
    return None
<|endoftext|># Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Contains a logger to push training logs to the Hub, using Tensorboard."""

from pathlib import Path

from ._commit_scheduler import CommitScheduler
from .errors import EntryNotFoundError
from .repocard import ModelCard
from .utils import experimental


def _load_summary_writer():
    """Load a SummaryWriter implementation only when an HFSummaryWriter is instantiated."""
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "You must have `tensorboard` installed to use `HFSummaryWriter`. Please run `pip install --upgrade"
                " tensorboardX` first."
            ) from None
    return SummaryWriter


class HFSummaryWriter:
    """
    Wrapper around the tensorboard's `SummaryWriter` to push training logs to the Hub.

    Data is logged locally and then pushed to the Hub asynchronously. Pushing data to the Hub is done in a separate
    thread to avoid blocking the training script. In particular, if the upload fails for any reason (e.g. a connection
    issue), the main script will not be interrupted. Data is automatically pushed to the Hub every `commit_every`
    minutes (default to every 5 minutes).

    > [!WARNING]
    > `HFSummaryWriter` is experimental. Its API is subject to change in the future without prior notice.

    Args:
        repo_id (`str`):
            The id of the repo to which the logs will be pushed.
        logdir (`str`, *optional*):
            The directory where the logs will be written. If not specified, a local directory will be created by the
            underlying `SummaryWriter` object.
        commit_every (`int` or `float`, *optional*):
            The frequency (in minutes) at which the logs will be pushed to the Hub. Defaults to 5 minutes.
        squash_history (`bool`, *optional*):
            Whether to squash the history of the repo after each commit. Defaults to `False`. Squashing commits is
            useful to avoid degraded performances on the repo when it grows too large.
        repo_type (`str`, *optional*):
            The type of the repo to which the logs will be pushed. Defaults to "model".
        repo_revision (`str`, *optional*):
            The revision of the repo to which the logs will be pushed. Defaults to "main".
        repo_private (`bool`, *optional*):
            Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
        path_in_repo (`str`, *optional*):
            The path to the folder in the repo where the logs will be pushed. Defaults to "tensorboard/".
        repo_allow_patterns (`list[str]` or `str`, *optional*):
            A list of patterns to include in the upload. Defaults to `"*.tfevents.*"`. Check out the
            [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder) for more details.
        repo_ignore_patterns (`list[str]` or `str`, *optional*):
            A list of patterns to exclude in the upload. Check out the
            [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder) for more details.
        token (`str`, *optional*):
            Authentication token. Will default to the stored token. See https://huggingface.co/settings/token for more
            details
        kwargs:
            Additional keyword arguments passed to `SummaryWriter`.

    Examples:
    ```diff
    # Taken from https://pytorch.org/docs/stable/tensorboard.html
    - from torch.utils.tensorboard import SummaryWriter
    + from huggingface_hub import HFSummaryWriter

    import numpy as np

    - writer = SummaryWriter()
    + writer = HFSummaryWriter(repo_id="username/my-trained-model")

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    ```

    ```py
    >>> from huggingface_hub import HFSummaryWriter

    # Logs are automatically pushed every 15 minutes (5 by default) + when exiting the context manager
    >>> with HFSummaryWriter(repo_id="test_hf_logger", commit_every=15) as logger:
    ...     logger.add_scalar("a", 1)
    ...     logger.add_scalar("b", 2)
    ```
    """

    @experimental
    def __new__(cls, *args, **kwargs) -> "HFSummaryWriter":
        return super().__new__(cls)

    def __init__(
        self,
        repo_id: str,
        *,
        logdir: str | None = None,
        commit_every: int | float = 5,
        squash_history: bool = False,
        repo_type: str | None = None,
        repo_revision: str | None = None,
        repo_private: bool | None = None,
        path_in_repo: str | None = "tensorboard",
        repo_allow_patterns: list[str] | str | None = "*.tfevents.*",
        repo_ignore_patterns: list[str] | str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        # Initialize SummaryWriter
        self._summary_writer = _load_summary_writer()(logdir=logdir, **kwargs)
        self.logdir = self._summary_writer.logdir

        # Check logdir has been correctly initialized and fail early otherwise. In practice, SummaryWriter takes care of it.
        if not isinstance(self.logdir, str):
            raise ValueError(f"`self.logdir` must be a string. Got '{self.logdir}' of type {type(self.logdir)}.")

        # Append logdir name to `path_in_repo`
        if path_in_repo is None or path_in_repo == "":
            path_in_repo = Path(self.logdir).name
        else:
            path_in_repo = path_in_repo.strip("/") + "/" + Path(self.logdir).name

        # Initialize scheduler
        self.scheduler = CommitScheduler(
            folder_path=self.logdir,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=repo_revision,
            private=repo_private,
            token=token,
            allow_patterns=repo_allow_patterns,
            ignore_patterns=repo_ignore_patterns,
            every=commit_every,
            squash_history=squash_history,
        )

        # Exposing some high-level info at root level
        self.repo_id = self.scheduler.repo_id
        self.repo_type = self.scheduler.repo_type
        self.repo_revision = self.scheduler.revision

        # Add `hf-summary-writer` tag to the model card metadata
        try:
            card = ModelCard.load(repo_id_or_path=self.repo_id, repo_type=self.repo_type)
        except EntryNotFoundError:
            card = ModelCard("")
        tags = card.data.get("tags", [])
        if "hf-summary-writer" not in tags:
            tags.append("hf-summary-writer")
            card.data["tags"] = tags
            card.push_to_hub(repo_id=self.repo_id, repo_type=self.repo_type)

    def __getattr__(self, name: str):
        """Forward the SummaryWriter API to the lazily-created writer."""
        return getattr(self._summary_writer, name)

    def __enter__(self) -> "HFSummaryWriter":
        self._summary_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Push to hub in a non-blocking way when exiting the logger's context manager."""
        self._summary_writer.__exit__(exc_type, exc_val, exc_tb)
        future = self.scheduler.trigger()
        future.result()
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
import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from typing import Any, BinaryIO, Callable, Iterator, List, Literal, overload
from urllib.parse import urlparse

import httpx2

from . import constants
from ._sandbox_cache import (
    HOST_TRUST_TTL,
    CacheContext,
    CachedHost,
    delete_pool_cache,
    read_pool_cache,
    save_pool_cache,
)
from ._space_api import Volume
from .errors import HfHubHTTPError, SandboxCommandError, SandboxError
from .hf_api import HfApi, JobInfo
from .utils import get_token, logging
from .utils._parsing import parse_duration


logger = logging.get_logger(__name__)

# Port the sandbox server listens on inside the job. Deliberately uncommon so that typical user ports stay free.
SANDBOX_SERVER_PORT = 49983

# Stable marker present on every sandbox job to ease filtering
SANDBOX_LABEL = "hf-sandbox"
# Sandbox mode: "dedicated" or "pool" to ease filtering
MODE_LABEL = "hf-sandbox-mode"
MODE_DEDICATED = "dedicated"
MODE_POOL = "pool"
# Pool name, to scope host reuse to a named group (see SandboxPool(name=...)). Present on pool
# host jobs only. Pool config (capacity, idle timeout) lives in the host's env vars, read via
# inspect_job — labels are kept for filtering/grouping only.
POOL_LABEL = "hf-sandbox-pool"
# Per-job public nonce the sandbox token is derived from (see _derive_sandbox_token), so
# `Sandbox.connect(id)` can recompute the token from any machine with no local state.
NONCE_LABEL = "hf-sandbox-nonce"

RESERVED_SANDBOX_LABELS = frozenset({SANDBOX_LABEL, MODE_LABEL, POOL_LABEL, NONCE_LABEL})

DEFAULT_IMAGE = "python:3.12"

# The exact sbx-server build every sandbox job downloads and runs as root, pinned by digest.
# The bucket object is fetched by its sha256 name and the download is verified before it is
# made executable (see `_BOOTSTRAP_DOWNLOAD`), so the bytes that run are the bytes that were
# reviewed for this release -- not whatever the mutable `sbx-server` alias points at today.
# Both constants move together on every server release; the digest is printed by the server
# repo's publish workflow.
SANDBOX_SERVER_VERSION = "0.7.0"
SANDBOX_SERVER_SHA256 = "eb2b04f79fbe765195300aaf1582be2a2a9ea698ff067d0cab428171d5584932"

# The sbx-server wire contract this client drives, checked against `/health`'s `protocol` on
# startup. A pool host keeps the binary it downloaded at boot for up to 24h, so pinning a
# digest does not stop this client from meeting a server it did not pin -- the check does.
SANDBOX_SERVER_PROTOCOL = 3

DEFAULT_IDLE_TIMEOUT = 10 * 60  # 10 minutes
SANDBOX_MAX_LIFETIME = "24h"

DEFAULT_SANDBOXES_PER_HOST = 50

# How long `close()` waits for in-flight creates before giving up on them. A
# wedged create must not hang process exit, but returning immediately would let
# a host it is still booting escape teardown.
_CLOSE_DRAIN_TIMEOUT = 30.0

# Ceiling on the output `run()` will accumulate for its result. Without one, a
# runaway command (a loop printing to stdout, say) is an unbounded allocation in
# the caller's process -- and it happened even when a callback was already
# consuming the output. Raising beats being OOM-killed with no explanation.
MAX_CAPTURED_OUTPUT_CHARS = 64 * 1024 * 1024

# Which pool hosts `create()` may adopt from job labels. Labels are set by whoever
# creates a Job, so "any job carrying our pool's labels" is not an identity claim --
# see the `adopt_hosts` argument of [`SandboxPool`].
ADOPT_OWN = "own"
ADOPT_NAMESPACE = "namespace"
ADOPT_NEVER = "never"

SHARED_ID_SEP = "."

# Job stages in which a sandbox/host is finished and needs no teardown.
_TERMINAL_STAGES = ("COMPLETED", "ERROR", "DELETED", "CANCELED")

# Safety bound on create()'s pack-retry loop
_MAX_PACK_ROUNDS = 8

# hf-mount path where the server bucket is mounted on every sandbox job
_SERVER_MOUNT_PATH = "/.hf-sbx-server"

# Job startup script (needs only /bin/sh). The server bucket is public, so the download is
# unauthenticated: no HF credential is ever placed in the job environment (see `_derive_sandbox_token`).
#
# It fetches a public object and runs it as root, as PID 1, holding the sandbox token -- so it
# checks the bytes against `SANDBOX_SERVER_SHA256` and refuses to run them if they don't match.
# The check happens *before* `chmod +x`: an unverified file that is already executable is one
# slip away from being executed. Neither `sha256sum` nor `openssl` is guaranteed to exist in an
# arbitrary image, so both are tried; with neither, the default is to refuse rather than to run
# unverified code, and `SBX_ALLOW_UNVERIFIED_SERVER=1` (dedicated job `env=` or the image environment) is the way out for an
# image that has to. It never bypasses a *failed* check, only a missing tool.
_BOOTSTRAP_DOWNLOAD = """\
set -e
d=/tmp/.sbx-server
if command -v wget >/dev/null 2>&1; then wget -q -O "$d" "$SBX_SERVER_URL"
elif command -v curl >/dev/null 2>&1; then curl -fsSL -o "$d" "$SBX_SERVER_URL"
else cp "$SBX_SERVER_MOUNT/sbx-server-$SBX_SERVER_SHA256" "$d"; fi
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$d" | cut -d' ' -f1)
elif command -v openssl >/dev/null 2>&1; then
  actual=$(openssl dgst -sha256 -r "$d" | cut -d' ' -f1)
elif [ "${SBX_ALLOW_UNVERIFIED_SERVER:-}" = 1 ]; then
  echo "sbx: SBX_ALLOW_UNVERIFIED_SERVER=1: running the sandbox server unverified" >&2
  actual=$SBX_SERVER_SHA256  # opted out, so there is nothing left to compare against
else
  echo "sbx: cannot verify the sandbox server digest: this image has neither sha256sum nor" >&2
  echo "sbx: openssl, so refusing to run it. Use an image with either hash tool installed." >&2
  echo "sbx: Dedicated jobs may opt out with env={'SBX_ALLOW_UNVERIFIED_SERVER': '1'};" >&2
  echo "sbx: pool hosts require setting it in the image environment instead." >&2
  exit 1
fi
if [ "$actual" != "$SBX_SERVER_SHA256" ]; then
  echo "sbx: sandbox server digest mismatch: got $actual," >&2
  echo "sbx: expected $SBX_SERVER_SHA256." >&2
  echo "sbx: refusing to run it. Upgrade huggingface_hub if this client is pinned to a" >&2
  echo "sbx: digest that is no longer published." >&2
  exit 1
fi
chmod +x "$d"
unset SBX_SERVER_URL SBX_SERVER_MOUNT SBX_SERVER_SHA256 SBX_ALLOW_UNVERIFIED_SERVER
exec "$d"
"""


def _derive_sandbox_token(hf_token: str, nonce: str) -> str:
    """Derive the sandbox auth token from the user's HF token and the job's nonce.

    Stateless: any machine holding the same HF token can recompute it from the
    nonce stored in the job's labels, so `Sandbox.connect(job_id)` needs no local state.
    Only the derived token is passed to the sandbox server as a job secret; the HF token
    itself is not.

    Scope: one nonce is minted per *job*, not per sandbox. For a dedicated sandbox those are
    the same thing. In a pool this derives the *host* credential, which manages the pool and
    can recover per-sandbox tokens -- a leak there is host-wide. Each pooled sandbox also has
    its own random capability token, minted by the host server, which is what per-sandbox
    operations and `proxy_headers` use.

    This is not a hardened boundary in either direction. An untrusted image can own the
    sandbox port, so don't treat it as a guarantee that credentials stay out of the sandbox;
    and the token is derived for whichever job carries the matching nonce label, so don't
    treat holding it as proof of which job you are talking to.
    """
    return hmac.new(hf_token.encode(), f"hf-sandbox:{nonce}".encode(), hashlib.sha256).hexdigest()


def _duration_to_secs(duration: int | float | str) -> int:
    """Parse a duration like 300, "300s", "10m", "2h", "1d" into seconds."""
    if isinstance(duration, (int, float)):
        return int(duration)
    return parse_duration(duration)


@dataclass
class SandboxCommandResult:
    """Result of a command executed in a sandbox with [`Sandbox.run`]."""

    exit_code: int | None
    stdout: str
    stderr: str
    signal: int | None = None
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def __repr__(self) -> str:
        out = self.stdout if len(self.stdout) <= 80 else self.stdout[:77] + "..."
        return f"SandboxCommandResult(exit_code={self.exit_code}, stdout={out!r}, duration_ms={self.duration_ms})"


@dataclass
class SandboxProcess:
    """A background process started in a sandbox with [`Sandbox.run`]`(..., background=True)`.

    List a sandbox's processes with [`Sandbox.processes`] and stop one with [`SandboxProcess.kill`].
    Recently completed processes stay in the listing (the server keeps a bounded number of
    them), so `running` and `exit_code` tell whether a process is still alive or already
    exited (as of when it was listed).
    """

    # Server-assigned handle, and the only identifier that addresses this process.
    # `pid` is the OS pid: useful for correlating with `ps` inside the sandbox, but
    # the OS may reuse it, so it is observational only. `None` for a process on a
    # host running a server that predates opaque ids.
    id: str | None
    pid: int
    cmd: str | List[str]
    # Back-reference to the sandbox, used by `kill()`. Excluded from repr/eq so a process
    # stays a plain data object.
    _sandbox: "Sandbox" = field(repr=False, compare=False)
    tag: str | None = None
    started_at_ms: int | None = None
    running: bool = True
    exit_code: int | None = None

    def kill(self) -> bool:
        """Terminate the background process. Idempotent.

        Returns whether this call is what stopped it: `False` means it had already
        exited or been terminated, which is not an error.

        Note that a descendant which detaches with `setsid()` leaves the signalled
        process group and outlives this call. Delete the sandbox to be certain
        everything it started is gone.
        """
        if self.id is None:
            raise SandboxError(
                "This process cannot be stopped individually: its sandbox runs a sandbox server "
                "that predates opaque process ids, so the server never issued one. Recycle the "
                "pool's hosts, or delete the sandbox to stop everything it started."
            )
        response = self._sandbox._request("DELETE", f"/processes/{self.id}")
        return bool(response.json().get("killed", False))


@dataclass
class FileEntry:
    """A file or directory inside a sandbox."""

    name: str
    path: str
    type: Literal["file", "dir", "symlink"]
    size: int
    mtime_ms: int | None = None
    mode: str = ""


@contextmanager
def _open_download_target(local_path: Path) -> Iterator[BinaryIO]:
    """Yield a writer whose bytes only appear at `local_path` once the download completes.

    Writing straight to the destination has three problems: an interrupted transfer leaves a
    truncated file at the final name; `open(path, "wb")` follows a symlink sitting there, so
    the sandbox's output can be redirected into any file this process can write; and a
    predictable temp name is a race another local process can win.

    So the bytes go to a temp file in the destination directory, created `O_EXCL` (nobody
    else's file) and `O_NOFOLLOW` (not a symlink) with mode `0600` and an unguessable name,
    then `os.replace`d into place. `rename` does not follow symlinks either, so a symlinked
    destination is *replaced* rather than written through, and a failure leaves nothing behind.
    """
    tmp = local_path.parent / f".{local_path.name}.{token_hex(8)}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp, local_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class SandboxFiles:
    """Filesystem operations inside a sandbox, available as [`Sandbox.files`].

    In shared (pool) mode, paths are rooted at the sandbox's private home — the
    only place its code can write — so a leading `/` is taken relative to that
    home. In dedicated mode, paths are absolute on the container filesystem.

    Paths are normalized lexically, then resolved by the host server relative to an open
    home-directory descriptor. The file API refuses symlinks on every component, including
    links whose targets are inside the same home.
    """

    # Above this size, transfers are split into ranged requests over parallel
    # connections: a single TCP stream through the jobs proxy is limited by the
    # bandwidth-delay product (~2 MiB/s at ~100ms RTT); parallel streams scale it.
    PARALLEL_THRESHOLD = 2 * 1024 * 1024
    PARALLEL_CHUNK_SIZE = 1 * 1024 * 1024
    PARALLEL_MAX_WORKERS = 16
    # Ceiling on what `read`/`read_text` will materialize in memory. A parallel
    # read used to collect every chunk into a list and *then* join it, so a 2 GB
    # file peaked at roughly twice its size before the caller saw a byte. Reading
    # a file into memory is inherently bounded by the file; this makes the bound
    # explicit and points at the streaming alternative instead of dying in an
    # allocator.
    MAX_READ_BYTES = 512 * 1024 * 1024

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> bytes:
        """Read a file from the sandbox and return its content as bytes.

        Raises [`SandboxError`] above `MAX_READ_BYTES`; use [`download`] for
        anything that large, which streams to disk instead of buffering.
        """
        size = self.stat(path).size
        if size > self.MAX_READ_BYTES:
            raise SandboxError(
                f"{path} is {size} bytes, over the {self.MAX_READ_BYTES}-byte limit for reading into "
                "memory. Use `files.download(path, local_path)`, which streams to disk."
            )
        if size > self.PARALLEL_THRESHOLD:
            # Write each range into one preallocated buffer as it arrives, rather
            # than collecting every chunk and joining: the join doubled peak memory.
            buffer = bytearray(size)
            for offset, part in self._read_ranges(path, size):
                buffer[offset : offset + len(part)] = part
            return bytes(buffer)
        return self._read_range(path, 0, size)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read a file from the sandbox and return its content as a string."""
        return self.read(path).decode(encoding)

    def write(self, path: str, data: str | bytes | BinaryIO, mode: str | None = None) -> None:
        """Write content to a file in the sandbox (parent directories are created)."""
        if isinstance(data, str):
            data = data.encode()
        elif not isinstance(data, bytes):
            data = data.read()  # binary file object
        if len(data) > self.PARALLEL_THRESHOLD:
            self._write_ranges(path, data, mode)
            return
        params = {"path": path}
        if mode is not None:
            params["mode"] = mode
        self._sandbox._request("PUT", "/files/write", params=params, content=data)

    def upload(self, local_path: str | Path, path: str, mode: str | None = None) -> None:
        """Upload a local file to the sandbox.

        The file is opened once and both sized (`fstat`) and read through that same
        descriptor, so what is uploaded is the file that was measured -- rather than
        whatever the path resolves to a moment later.
        """
        with open(local_path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size > self.PARALLEL_THRESHOLD:
                # Each worker reads its own range straight off the descriptor, so
                # a large upload no longer means holding the whole file in memory
                # (`f.read()` on a 10 GB file was 10 GB of resident bytes).
                self._write_ranges_from_file(path, f, size, mode)
            else:
                self.write(path, f, mode=mode)

    def download(self, path: str, local_path: str | Path) -> None:
        """Download a file from the sandbox.

        The bytes land in a private temp file next to the destination and are renamed
        over it at the end, so an interrupted download leaves no truncated file and a
        symlink sitting at `local_path` is replaced rather than followed.
        """
        size = self.stat(path).size
        with _open_download_target(Path(local_path)) as f:
            if size > self.PARALLEL_THRESHOLD:
                # Seek-and-write per range as it completes, so peak memory is a
                # few chunks rather than the whole file.
                for offset, part in self._read_ranges(path, size):
                    f.seek(offset)
                    f.write(part)
                return
            with self._sandbox._stream("GET", "/files/read", params={"path": path}) as response:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

    def _ranges(self, size: int) -> List[tuple[int, int]]:
        chunk = self.PARALLEL_CHUNK_SIZE
        return [(offset, min(chunk, size - offset)) for offset in range(0, size, chunk)]

    def _parallel(self, items: List[Any], fn: Callable[[Any], Any]) -> List[Any]:
        """Run `fn(item)` over items concurrently.

        All workers share the sandbox's `httpx2.Client`, which is thread-safe and pools
        connections, so parallel transfers fan out over several streams at once.
        """
        workers = min(self.PARALLEL_MAX_WORKERS, len(items))
        with ThreadPoolExecutor(workers) as executor:
            return list(executor.map(fn, items))

    def _read_range(self, path: str, offset: int, length: int) -> bytes:
        data = bytearray()
        with self._sandbox._stream(
            "GET", "/files/read", params={"path": path, "offset": offset, "length": length}
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if len(data) + len(chunk) > length:
                    raise SandboxError("File response exceeded the requested range.")
                data.extend(chunk)
        if len(data) != length:
            raise SandboxError("File changed size during transfer; retry the download.")
        return bytes(data)

    def _read_ranges(self, path: str, size: int) -> Iterator[tuple[int, bytes]]:
        """Fetch every range in parallel, yielding `(offset, bytes)` as they land.

        Yielding rather than returning a list is the point: the caller places each
        chunk and drops it, so peak memory is a few chunks instead of the whole
        file plus a copy.
        """

        def fetch(rng: tuple[int, int]) -> tuple[int, bytes]:
            offset, length = rng
            return offset, self._read_range(path, offset, length)

        ranges = self._ranges(size)
        workers = min(self.PARALLEL_MAX_WORKERS, len(ranges))
        with ThreadPoolExecutor(workers) as executor:
            for start in range(0, len(ranges), workers):
                yield from executor.map(fetch, ranges[start : start + workers])

    def _write_ranges(self, path: str, data: bytes, mode: str | None) -> None:
        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            self._put_range(path, offset, data[offset : offset + length], mode, total=len(data))

        self._parallel(self._ranges(len(data)), push)

    def _write_ranges_from_file(self, path: str, source: BinaryIO, size: int, mode: str | None) -> None:
        """Upload `source` in parallel ranges, reading each range on demand.

        Serialize reads from the shared descriptor; network writes still overlap.
        """
        read_lock = threading.Lock()

        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            with read_lock:
                source.seek(offset)
                chunk = source.read(length)
            if len(chunk) != length:
                raise SandboxError("Upload source changed size during transfer.")
            self._put_range(path, offset, chunk, mode, total=size)

        self._parallel(self._ranges(size), push)

    def _put_range(self, path: str, offset: int, chunk: bytes, mode: str | None, *, total: int) -> None:
        params: dict[str, Any] = {"path": path, "offset": offset}
        if mode is not None:
            params["mode"] = mode
        # Tell the server the intended final size. A ranged write does not
        # truncate, so overwriting a larger file with a smaller one used to leave
        # the old bytes past the new end.
        if offset + len(chunk) >= total:
            params["truncate_to"] = total
        self._sandbox._request("PUT", "/files/write", params=params, content=chunk)

    def list(self, path: str) -> List[FileEntry]:
        """List a directory in the sandbox.

        The server paginates, so this follows the cursor and returns everything --
        a caller sees one list regardless of how the directory is chunked on the
        wire.
        """
        entries: List[FileEntry] = []
        after: str | None = None
        while True:
            params: dict[str, Any] = {"path": path}
            if after is not None:
                params["after"] = after
            payload = self._sandbox._request("GET", "/files/list", params=params).json()
            entries.extend(FileEntry(**entry) for entry in payload["entries"])
            after = payload.get("next")
            if not after:
                return entries

    def stat(self, path: str) -> FileEntry:
        """Get metadata of a file or directory in the sandbox."""
        response = self._sandbox._request("GET", "/files/stat", params={"path": path})
        return FileEntry(**response.json())

    def exists(self, path: str) -> bool:
        """Check whether a path exists in the sandbox."""
        try:
            self.stat(path)
            return True
        except SandboxError as e:
            if e.status_code == 404:
                return False
            raise  # network/auth/server errors must not be silently reported as "missing"

    def delete(self, path: str, recursive: bool = False) -> None:
        """Delete a file or directory in the sandbox."""
        params = {"path": path}
        if recursive:
            params["recursive"] = "1"
        self._sandbox._request("DELETE", "/files/delete", params=params)

    def mkdir(self, path: str) -> None:
        """Create a directory (and parents) in the sandbox."""
        self._sandbox._request("POST", "/files/mkdir", params={"path": path})


def _exec_payload(cmd: str | List[str], shell: bool | None) -> dict[str, Any]:
    """Build the `cmd`/`shell` part of an `/exec` payload, validating their consistency."""
    if shell is True and not isinstance(cmd, str):
        raise ValueError("shell=True requires `cmd` to be a shell command string, not a list.")
    if shell is False and isinstance(cmd, str):
        raise ValueError("shell=False requires `cmd` to be an argv list (e.g. ['echo', 'hi']), not a string.")
    payload: dict[str, Any] = {"cmd": cmd}
    if shell is not None:
        payload["shell"] = shell
    return payload


def _iter_events(response: httpx2.Response) -> Iterator[dict]:
    """Iterate NDJSON events from a streaming response, skipping keepalive pings."""
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") != "ping":
            yield event


def _raise_for_status(response: httpx2.Response) -> None:
    """Read the error body and raise a SandboxError (works for streaming responses too)."""
    response.read()  # no-op for buffered responses, reads the body for streaming ones
    try:
        message = response.json()["error"]
    except Exception:
        message = response.text[:500]
    raise SandboxError(f"Sandbox API error ({response.status_code}): {message}", status_code=response.status_code)


def _check_server_protocol(response: httpx2.Response, job_id: str) -> None:
    """Refuse a server whose wire contract this client cannot drive.

    Permissive in one direction only. A *newer* server is accepted: it declares its own
    backwards compatibility by keeping the protocol integer it still supports. An *older* one
    is refused, because the mismatches are things like the per-sandbox capability token, which
    surfaces as a 403 on an unrelated route several calls later instead of here.

    `version` is no use for this: it moves for a doc fix as readily as for a protocol break.
    """
    try:
        protocol = response.json().get("protocol")
    except Exception:
        protocol = None
    if isinstance(protocol, int) and not isinstance(protocol, bool) and protocol >= SANDBOX_SERVER_PROTOCOL:
        return
    # No field at all means a server published before protocols were declared, i.e. 1.
    reported = protocol if protocol is not None else "1 (declares no protocol)"
    raise SandboxError(
        f"Sandbox job {job_id} runs sbx-server protocol {reported}, but this version of "
        f"huggingface_hub needs protocol {SANDBOX_SERVER_PROTOCOL} or newer. A pool host keeps the "
        "server binary it downloaded at boot for up to 24h, so recycle this pool's hosts (kill them, "
        "or let them idle-time-out) to pick up the current server."
    )


class _SandboxAuth(httpx2.Auth):
    """Attach the *current* HF bearer to every request.

    The bearer used to be captured once when the transport was built. A sandbox
    host lives up to 24h, so an OIDC/OAuth credential can expire or rotate inside
    that window, after which every request to the Jobs proxy failed even though
    the handle was still perfectly good. Reading it per request fixes that.

    The `X-Sandbox-Token` is deliberately *not* re-derived: the server holds the
    value derived from the bearer that created the job, so re-deriving from a
    rotated bearer would produce a token the server has never seen. The two
    credentials answer different gates -- the proxy wants a live bearer, the
    server wants the original capability -- and only the first needs refreshing.
    """

    def __init__(self, api: HfApi) -> None:
        self._api = api

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {_effective_token(self._api)}"
        yield request


def _require_scoped_token(token: Any) -> str:
