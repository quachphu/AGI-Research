        "automatic-speech-recognition": ReplicateAutomaticSpeechRecognitionTask(),
        "image-to-image": ReplicateImageToImageTask(),
        "text-to-image": ReplicateTextToImageTask(),
        "text-to-speech": ReplicateTextToSpeechTask(),
        "text-to-video": ReplicateTask("text-to-video"),
    },
    "scaleway": {
        "conversational": ScalewayConversationalTask(),
        "feature-extraction": ScalewayFeatureExtractionTask(),
    },
    "together": {
        "conversational": TogetherConversationalTask(),
        "feature-extraction": TogetherFeatureExtractionTask(),
        "image-to-image": TogetherImageToImageTask(),
        "image-to-video": TogetherImageToVideoTask(),
        "text-generation": TogetherTextGenerationTask(),
        "text-to-image": TogetherTextToImageTask(),
        "text-to-speech": TogetherTextToSpeechTask(),
        "text-to-video": TogetherTextToVideoTask(),
    },
    "wavespeed": {
        "text-to-image": WavespeedAITextToImageTask(),
        "text-to-video": WavespeedAITextToVideoTask(),
        "image-to-image": WavespeedAIImageToImageTask(),
        "image-to-video": WavespeedAIImageToVideoTask(),
    },
    "zai-org": {
        "conversational": ZaiConversationalTask(),
        "text-to-image": ZaiTextToImageTask(),
    },
}


def get_provider_helper(provider: PROVIDER_OR_POLICY_T | None, task: str, model: str | None) -> TaskProviderHelper:
    """Get provider helper instance by name and task.

    Args:
        provider (`str`, *optional*): name of the provider, or "auto" to automatically select the provider for the model.
        task (`str`): Name of the task
        model (`str`, *optional*): Name of the model
    Returns:
        TaskProviderHelper: Helper instance for the specified provider and task

    Raises:
        ValueError: If provider or task is not supported
    """

    if (model is None and provider in (None, "auto")) or (
        model is not None and model.startswith(("http://", "https://"))
    ):
        provider = "hf-inference"

    if provider is None:
        logger.info(
            "No provider specified for task `conversational`. Defaulting to server-side auto routing."
            if task == "conversational"
            else "Defaulting to 'auto' which will select the first provider available for the model, sorted by the user's order in https://hf.co/settings/inference-providers."
        )
        provider = "auto"

    if provider == "auto":
        if model is None:
            raise ValueError("Specifying a model is required when provider is 'auto'")
        if task == "conversational":
            # Special case: we have a dedicated auto-router for conversational models. No need to fetch provider mapping.
            return CONVERSATIONAL_AUTO_ROUTER

        provider_mapping = _fetch_inference_provider_mapping(model)
        provider = next(iter(provider_mapping)).provider

    provider_tasks = PROVIDERS.get(provider)  # type: ignore
    if provider_tasks is None:
        raise ValueError(
            f"Provider '{provider}' not supported. Available values: 'auto' or any provider from {list(PROVIDERS.keys())}."
            "Passing 'auto' (default value) will automatically select the first provider available for the model, sorted "
            "by the user's order in https://hf.co/settings/inference-providers."
        )

    if task not in provider_tasks:
        raise ValueError(
            f"Task '{task}' not supported for provider '{provider}'. Available tasks: {list(provider_tasks.keys())}"
        )
    return provider_tasks[task]
<|endoftext|>from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
)
from huggingface_hub.utils import get_session


_PROVIDER = "novita"
_BASE_URL = "https://api.novita.ai"


class NovitaTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        # there is no v1/ route for novita
        return "/v3/openai/completions"

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class NovitaConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        # there is no v1/ route for novita
        return "/v3/openai/chat/completions"


class NovitaTextToVideoTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-video")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"/v3/hf/{mapped_model}"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {"prompt": inputs, **filter_none(parameters)}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        if not (
            isinstance(response_dict, dict)
            and "video" in response_dict
            and isinstance(response_dict["video"], dict)
            and "video_url" in response_dict["video"]
        ):
            raise ValueError("Expected response format: { 'video': { 'video_url': string } }")

        video_url = response_dict["video"]["video_url"]
        return get_session().get(video_url).content
<|endoftext|>import base64
import time
from abc import ABC
from typing import Any
from urllib.parse import urlparse

from huggingface_hub import constants
from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict, _as_url
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session, hf_raise_for_status
from huggingface_hub.utils.logging import get_logger


logger = get_logger(__name__)

# Arbitrary polling interval
_POLLING_INTERVAL = 0.5


class FalAITask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider="fal-ai", base_url="https://fal.run", task=task)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        if not api_key.startswith("hf_"):
            headers["authorization"] = f"Key {api_key}"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"/{mapped_model}"


class FalAIQueueTask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider="fal-ai", base_url="https://queue.fal.run", task=task)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        if not api_key.startswith("hf_"):
            headers["authorization"] = f"Key {api_key}"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        if api_key.startswith("hf_"):
            # Use the queue subdomain for HF routing
            return f"/{mapped_model}?_subdomain=queue"
        return f"/{mapped_model}"

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        response_dict = _as_dict(response)

        request_id = response_dict.get("request_id")
        if not request_id:
            raise ValueError("No request ID found in the response")
        if request_params is None:
            raise ValueError(
                f"A `RequestParameters` object should be provided to get {self.task} responses with Fal AI."
            )

        # extract the base url and query params
        parsed_url = urlparse(request_params.url)
        # a bit hacky way to concatenate the provider name without parsing `parsed_url.path`
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{'/fal-ai' if parsed_url.netloc == 'router.huggingface.co' else ''}"
        query_param = f"?{parsed_url.query}" if parsed_url.query else ""

        # extracting the provider model id for status and result urls
        # from the response as it might be different from the mapped model in `request_params.url`
        model_id = urlparse(response_dict.get("response_url")).path
        status_url = f"{base_url}{str(model_id)}/status{query_param}"
        result_url = f"{base_url}{str(model_id)}{query_param}"

        status = response_dict.get("status")
        logger.info("Generating the output.. this can take several minutes.")
        while status != "COMPLETED":
            time.sleep(_POLLING_INTERVAL)
            status_response = get_session().get(status_url, headers=request_params.headers)
            hf_raise_for_status(status_response)
            status = status_response.json().get("status")

        return get_session().get(result_url, headers=request_params.headers).json()


class FalAIAutomaticSpeechRecognitionTask(FalAITask):
    def __init__(self):
        super().__init__("automatic-speech-recognition")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        if isinstance(inputs, str) and inputs.startswith(("http://", "https://")):
            # If input is a URL, pass it directly
            audio_url = inputs
        else:
            # If input is a file path, read it first
            if isinstance(inputs, str):
                with open(inputs, "rb") as f:
                    inputs = f.read()

            audio_b64 = base64.b64encode(inputs).decode()
            content_type = "audio/mpeg"
            audio_url = f"data:{content_type};base64,{audio_b64}"

        return {"audio_url": audio_url, **filter_none(parameters)}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        text = _as_dict(response)["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from FalAI API. Expected string, got {type(text)}.")
        return {"text": text}


class FalAITextToImageTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload: dict[str, Any] = {
            "prompt": inputs,
            **filter_none(parameters),
        }
        if "width" in payload and "height" in payload:
            payload["image_size"] = {
                "width": payload.pop("width"),
                "height": payload.pop("height"),
            }
        if provider_mapping_info.adapter_weights_path is not None:
            lora_path = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
                repo_id=provider_mapping_info.hf_model_id,
                revision="main",
                filename=provider_mapping_info.adapter_weights_path,
            )
            payload["loras"] = [{"path": lora_path, "scale": 1}]
            if provider_mapping_info.provider_id == "fal-ai/lora":
                # little hack: fal requires the base model for stable-diffusion-based loras but not for flux-based
                # See payloads in https://fal.ai/models/fal-ai/lora/api vs https://fal.ai/models/fal-ai/flux-lora/api
                payload["model_name"] = "stabilityai/stable-diffusion-xl-base-1.0"

        return payload

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        url = _as_dict(response)["images"][0]["url"]
        return get_session().get(url).content


class FalAITextToSpeechTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {"text": inputs, **filter_none(parameters)}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        url = _as_dict(response)["audio"]["url"]
        return get_session().get(url).content


class FalAITextToVideoTask(FalAIQueueTask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload: dict[str, Any] = {"prompt": inputs, **filter_none(parameters)}
        if provider_mapping_info.adapter_weights_path is not None:
            lora_path = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
                repo_id=provider_mapping_info.hf_model_id,
                revision="main",
                filename=provider_mapping_info.adapter_weights_path,
            )
            payload["loras"] = [{"path": lora_path, "scale": 1}]
        return payload

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        output = super().get_response(response, request_params)
        url = _as_dict(output)["video"]["url"]
        return get_session().get(url).content


class FalAIImageToImageTask(FalAIQueueTask):
    def __init__(self):
        super().__init__("image-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        image_url = _as_url(inputs, default_mime_type="image/jpeg")
        if "target_size" in parameters:
            parameters["image_size"] = parameters.pop("target_size")
        payload: dict[str, Any] = {
            "image_url": image_url,
            "image_urls": [image_url],
            **filter_none(parameters),
        }
        if provider_mapping_info.adapter_weights_path is not None:
            lora_path = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
                repo_id=provider_mapping_info.hf_model_id,
                revision="main",
                filename=provider_mapping_info.adapter_weights_path,
            )
            payload["loras"] = [{"path": lora_path, "scale": 1}]

        return payload

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        output = super().get_response(response, request_params)
        url = _as_dict(output)["images"][0]["url"]
        return get_session().get(url).content


class FalAIImageToVideoTask(FalAIQueueTask):
    def __init__(self):
        super().__init__("image-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        image_url = _as_url(inputs, default_mime_type="image/jpeg")
        payload: dict[str, Any] = {
            "image_url": image_url,
            **filter_none(parameters),
        }
        if provider_mapping_info.adapter_weights_path is not None:
            lora_path = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
                repo_id=provider_mapping_info.hf_model_id,
                revision="main",
                filename=provider_mapping_info.adapter_weights_path,
            )
            payload["loras"] = [{"path": lora_path, "scale": 1}]
        return payload

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        output = super().get_response(response, request_params)
        url = _as_dict(output)["video"]["url"]
        return get_session().get(url).content


class FalAIImageSegmentationTask(FalAIQueueTask):
    def __init__(self):
        super().__init__("image-segmentation")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        image_url = _as_url(inputs, default_mime_type="image/png")
        payload: dict[str, Any] = {
            "image_url": image_url,
            **filter_none(parameters),
            "sync_mode": True,
        }
        return payload

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        result = super().get_response(response, request_params)
        result_dict = _as_dict(result)

        if "image" not in result_dict:
            raise ValueError(f"Response from fal ai image-segmentation API does not contain an image: {result_dict}")

        image_data = result_dict["image"]
        if "url" not in image_data:
            raise ValueError(f"Image data from fal ai image-segmentation API does not contain a URL: {image_data}")

        image_url = image_data["url"]

        if isinstance(image_url, str) and image_url.startswith("data:"):
            if "," in image_url:
                mask_base64 = image_url.split(",", 1)[1]
            else:
                raise ValueError(f"Invalid data URL format: {image_url}")
        else:
            # or it's a regular URL, fetch it
            mask_response = get_session().get(image_url)
            hf_raise_for_status(mask_response)
            mask_base64 = base64.b64encode(mask_response.content).decode()

        return [
            {
                "label": "mask",
                "mask": mask_base64,
            }
        ]
<|endoftext|>import base64
import time
from abc import ABC
from typing import Any
from urllib.parse import urlparse

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session, hf_raise_for_status
from huggingface_hub.utils.logging import get_logger


logger = get_logger(__name__)

# Polling interval (in seconds)
_POLLING_INTERVAL = 0.5


class WavespeedAITask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider="wavespeed", base_url="https://api.wavespeed.ai", task=task)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"/api/v3/{mapped_model}"

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        response_dict = _as_dict(response)
        data = response_dict.get("data", {})
        result_path = data.get("urls", {}).get("get")

        if not result_path:
            raise ValueError("No result URL found in the response")
        if request_params is None:
            raise ValueError("A `RequestParameters` object should be provided to get responses with WaveSpeed AI.")

        # Parse the request URL to determine base URL
        parsed_url = urlparse(request_params.url)
        # Add /wavespeed to base URL if going through HF router
        if parsed_url.netloc == "router.huggingface.co":
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/wavespeed"
        else:
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Extract path from result_path URL
        if isinstance(result_path, str):
            result_url_path = urlparse(result_path).path
        else:
            result_url_path = result_path

        result_url = f"{base_url}{result_url_path}"

        logger.info("Processing request, polling for results...")

        # Poll until task is completed
        while True:
            time.sleep(_POLLING_INTERVAL)
            result_response = get_session().get(result_url, headers=request_params.headers)
            hf_raise_for_status(result_response)

            result = result_response.json()
            task_result = result.get("data", {})
            status = task_result.get("status")

            if status == "completed":
                # Get content from the first output URL
                if not task_result.get("outputs") or len(task_result["outputs"]) == 0:
                    raise ValueError("No output URL in completed response")

                output_url = task_result["outputs"][0]
                return get_session().get(output_url).content
            elif status == "failed":
                error_msg = task_result.get("error", "Task failed with no specific error message")
                raise ValueError(f"WaveSpeed AI task failed: {error_msg}")
            elif status in ["processing", "created"]:
                continue
            else:
                raise ValueError(f"Unknown status: {status}")


class WavespeedAITextToImageTask(WavespeedAITask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        return {"prompt": inputs, **filter_none(parameters)}


class WavespeedAITextToVideoTask(WavespeedAITextToImageTask):
    def __init__(self):
        WavespeedAITask.__init__(self, "text-to-video")


class WavespeedAIImageToImageTask(WavespeedAITask):
    def __init__(self):
        super().__init__("image-to-image")

    def _prepare_payload_as_dict(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        # Convert inputs to image (URL or base64)
        if isinstance(inputs, str) and inputs.startswith(("http://", "https://")):
            image = inputs
        elif isinstance(inputs, str):
            # If input is a file path, read it first
            with open(inputs, "rb") as f:
                file_content = f.read()
            image_b64 = base64.b64encode(file_content).decode("utf-8")
            image = f"data:image/jpeg;base64,{image_b64}"
        else:
            # If input is binary data
            image_b64 = base64.b64encode(inputs).decode("utf-8")
            image = f"data:image/jpeg;base64,{image_b64}"

        # Extract prompt from parameters if present
        prompt = parameters.pop("prompt", None)
        payload = {"image": image, **filter_none(parameters)}
        if prompt is not None:
            payload["prompt"] = prompt

        return payload


class WavespeedAIImageToVideoTask(WavespeedAIImageToImageTask):
    def __init__(self):
        WavespeedAITask.__init__(self, "image-to-video")
<|endoftext|>from huggingface_hub.inference._providers._common import BaseConversationalTask


_PROVIDER = "ovhcloud"
_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net"


class OVHcloudConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
<|endoftext|>from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict, _as_url
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session


_PROVIDER = "replicate"
_BASE_URL = "https://api.replicate.com"


class ReplicateTask(TaskProviderHelper):
    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        headers["Prefer"] = "wait"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        if ":" in mapped_model:
            return "/v1/predictions"
        return f"/v1/models/{mapped_model}/predictions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        payload: dict[str, Any] = {"input": {"prompt": inputs, **filter_none(parameters)}}
        if ":" in mapped_model:
            version = mapped_model.split(":", 1)[1]
            payload["version"] = version
        return payload

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        if response_dict.get("output") is None:
            raise TimeoutError(
                f"Inference request timed out after 60 seconds. No output generated for model {response_dict.get('model')}"
                "The model might be in cold state or starting up. Please try again later."
            )
        output_url = (
            response_dict["output"] if isinstance(response_dict["output"], str) else response_dict["output"][0]
        )
        return get_session().get(output_url).content


class ReplicateTextToImageTask(ReplicateTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload: dict = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)  # type: ignore
        if provider_mapping_info.adapter_weights_path is not None:
            payload["input"]["lora_weights"] = f"https://huggingface.co/{provider_mapping_info.hf_model_id}"
        return payload


class ReplicateTextToSpeechTask(ReplicateTask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload: dict = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)  # type: ignore
        payload["input"]["text"] = payload["input"].pop("prompt")  # rename "prompt" to "text" for TTS
        return payload


class ReplicateAutomaticSpeechRecognitionTask(ReplicateTask):
    def __init__(self) -> None:
        super().__init__("automatic-speech-recognition")

    def _prepare_payload_as_dict(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        audio_url = _as_url(inputs, default_mime_type="audio/wav")

        payload: dict[str, Any] = {
            "input": {
                **{"audio": audio_url},
                **filter_none(parameters),
            }
        }

        if ":" in mapped_model:
            payload["version"] = mapped_model.split(":", 1)[1]

        return payload

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        output = response_dict.get("output")

        if isinstance(output, str):
            return {"text": output}

        if isinstance(output, list) and output:
            first_item = output[0]
            if isinstance(first_item, str):
                return {"text": first_item}
            if isinstance(first_item, dict):
                output = first_item

        text: str | None = None
        if isinstance(output, dict):
            transcription = output.get("transcription")
            if isinstance(transcription, str):
                text = transcription

            translation = output.get("translation")
            if isinstance(translation, str):
                text = translation

            txt_file = output.get("txt_file")
            if isinstance(txt_file, str):
                text_response = get_session().get(txt_file)
                text_response.raise_for_status()
                text = text_response.text

        if text is not None:
            return {"text": text}

        raise ValueError("Received malformed response from Replicate automatic-speech-recognition API")


class ReplicateImageToImageTask(ReplicateTask):
    def __init__(self):
        super().__init__("image-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        image_url = _as_url(inputs, default_mime_type="image/jpeg")

        # Different Replicate models expect the image in different keys
        payload: dict[str, Any] = {
            "input": {
                "image": image_url,
                "images": [image_url],
                "input_image": image_url,
                "input_images": [image_url],
                **filter_none(parameters),
            }
        }

        mapped_model = provider_mapping_info.provider_id
        if ":" in mapped_model:
            version = mapped_model.split(":", 1)[1]
            payload["version"] = version
        return payload
<|endoftext|>import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from huggingface_hub import constants
from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import (
    MimeBytes,
    RequestParameters,
    _b64_encode,
    _bytes_to_dict,
    _open_as_mime_bytes,
)
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import build_hf_headers, get_session, get_token, hf_raise_for_status


class HFInferenceTask(TaskProviderHelper):
    """Base class for HF Inference API tasks."""

    def __init__(self, task: str):
        super().__init__(
            provider="hf-inference",
            base_url=constants.INFERENCE_PROXY_TEMPLATE.format(provider="hf-inference"),
            task=task,
        )

    def _prepare_api_key(self, api_key: str | None) -> str:
        # special case: for HF Inference we allow not providing an API key
        return api_key or get_token()  # type: ignore

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        if model is not None and model.startswith(("http://", "https://")):
            return InferenceProviderMapping(
                provider="hf-inference", providerId=model, hf_model_id=model, task=self.task, status="live"
            )
        model_id = model if model is not None else _fetch_recommended_models().get(self.task)
        if model_id is None:
            raise ValueError(
                f"Task {self.task} has no recommended model for HF Inference. Please specify a model"
                " explicitly. Visit https://huggingface.co/tasks for more info."
            )
        _check_supported_task(model_id, self.task)
        return InferenceProviderMapping(
            provider="hf-inference", providerId=model_id, hf_model_id=model_id, task=self.task, status="live"
        )

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        # hf-inference provider can handle URLs (e.g. Inference Endpoints or TGI deployment)
        if mapped_model.startswith(("http://", "https://")):
            return mapped_model
        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{self.base_url}/models/{mapped_model}/pipeline/{self.task}"
            if self.task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{self.base_url}/models/{mapped_model}"
        )

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")
        return filter_none({"inputs": inputs, "parameters": parameters})


class HFInferenceBinaryInputTask(HFInferenceTask):
    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return None

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        parameters = filter_none(parameters)
        extra_payload = extra_payload or {}
        has_parameters = len(parameters) > 0 or len(extra_payload) > 0

        # Raise if not a binary object or a local path or a URL.
        if not isinstance(inputs, (bytes, Path)) and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")

        # Send inputs as raw content when no parameters are provided
        if not has_parameters:
            return _open_as_mime_bytes(inputs)

        # Otherwise encode as b64
        return MimeBytes(
            json.dumps({"inputs": _b64_encode(inputs), "parameters": parameters, **extra_payload}).encode("utf-8"),
            mime_type="application/json",
        )


class HFInferenceConversational(HFInferenceTask):
    def __init__(self):
        super().__init__("conversational")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = filter_none(parameters)
        mapped_model = provider_mapping_info.provider_id
        payload_model = parameters.get("model") or mapped_model

        if payload_model is None or payload_model.startswith(("http://", "https://")):
            payload_model = "dummy"

        response_format = parameters.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            payload["response_format"] = {
                "type": "json_object",
                "value": response_format["json_schema"]["schema"],
            }
        # Same ordering rationale as `BaseConversationalTask`: `model` first so it can be read from a
        # small prefix of the body, and dropped from the `payload` spread so the already-resolved
        # `payload_model` wins.
        return {
            "model": payload_model,
            "messages": inputs,
            **{key: value for key, value in payload.items() if key != "model"},
        }

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        base_url = (
            mapped_model
            if mapped_model.startswith(("http://", "https://"))
            else f"{constants.INFERENCE_PROXY_TEMPLATE.format(provider='hf-inference')}/models/{mapped_model}"
        )
        return _build_chat_completion_url(base_url)


def _build_chat_completion_url(model_url: str) -> str:
    parsed = urlparse(model_url)
    path = parsed.path.rstrip("/")

    # If the path already ends with /chat/completions, we're done!
    if path.endswith("/chat/completions"):
        return model_url

    # Append /chat/completions if not already present
    if path.endswith("/v1"):
        new_path = path + "/chat/completions"
    # If path was empty or just "/", set the full path
    elif not path:
        new_path = "/v1/chat/completions"
    # Append /v1/chat/completions if not already present
    else:
        new_path = path + "/v1/chat/completions"

    # Reconstruct the URL with the new path and original query parameters.
    new_parsed = parsed._replace(path=new_path)
    return str(urlunparse(new_parsed))


@lru_cache(maxsize=1)
def _fetch_recommended_models() -> dict[str, str | None]:
    response = get_session().get(f"{constants.ENDPOINT}/api/tasks", headers=build_hf_headers())
    hf_raise_for_status(response)
    return {task: next(iter(details["widgetModels"]), None) for task, details in response.json().items()}


@lru_cache(maxsize=None)
def _check_supported_task(model: str, task: str) -> None:
    from huggingface_hub.hf_api import HfApi

    model_info = HfApi().model_info(model)
    pipeline_tag = model_info.pipeline_tag
    tags = model_info.tags or []
    if task in ("text-generation", "conversational"):
        if pipeline_tag == "text-generation":
            return

    if pipeline_tag == "text2text-generation":
        if task == "text-generation":
            return
        raise ValueError(f"Model '{model}' doesn't support task '{task}'.")

    if pipeline_tag == "image-text-to-text":
        if task == "conversational":
            return
        raise ValueError("Non-conversational image-text-to-text task is not supported.")

    if (
        task in ("feature-extraction", "sentence-similarity")
        and pipeline_tag in ("feature-extraction", "sentence-similarity")
        and task in tags
    ):
        # feature-extraction and sentence-similarity are interchangeable for HF Inference
        return

    # For all other tasks, just check pipeline tag
    if pipeline_tag != task:
        raise ValueError(
            f"Model '{model}' doesn't support task '{task}'. Supported tasks: '{pipeline_tag}', got: '{task}'"
        )
    return


class HFInferenceFeatureExtractionTask(HFInferenceTask):
    def __init__(self):
        super().__init__("feature-extraction")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")

        # Parameters are sent at root-level for feature-extraction task
        # See specs: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/tasks/feature-extraction/spec/input.json
        return {"inputs": inputs, **filter_none(parameters)}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            return _bytes_to_dict(response)
        return response
<|endoftext|>from functools import lru_cache
from typing import Any, overload

from huggingface_hub import constants
from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import MimeBytes, RequestParameters
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionInputMessage
from huggingface_hub.utils import build_hf_headers, get_token, logging


logger = logging.get_logger(__name__)


# Dev purposes only.
# If you want to try to run inference for a new model locally before it's registered on huggingface.co
# for a given Inference Provider, you can add it to the following dictionary.
HARDCODED_MODEL_INFERENCE_MAPPING: dict[str, dict[str, InferenceProviderMapping]] = {
    # "HF model ID" => InferenceProviderMapping object initialized with "Model ID on Inference Provider's side"
    #
    # Example:
    # "Qwen/Qwen2.5-Coder-32B-Instruct": InferenceProviderMapping(hf_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #                                    provider_id="Qwen2.5-Coder-32B-Instruct",
    #                                    task="conversational",
    #                                    status="live")
    "cerebras": {},
    "cohere": {},
    "deepinfra": {},
    "fal-ai": {},
    "fireworks-ai": {},
    "groq": {},
    "hf-inference": {},
    "nscale": {},
    "ovhcloud": {},
    "replicate": {},
    "scaleway": {},
    "together": {},
    "wavespeed": {},
    "zai-org": {},
}


@overload
def filter_none(obj: dict[str, Any]) -> dict[str, Any]: ...
@overload
def filter_none(obj: list[Any]) -> list[Any]: ...


def filter_none(obj: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                v = filter_none(v)
            cleaned[k] = v
        return cleaned

    if isinstance(obj, list):
        return [filter_none(v) if isinstance(v, (dict, list)) else v for v in obj]

    raise ValueError(f"Expected dict or list, got {type(obj)}")


class TaskProviderHelper:
    """Base class for task-specific provider helpers."""

    def __init__(self, provider: str, base_url: str, task: str) -> None:
        self.provider = provider
        self.task = task
        self.base_url = base_url

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: dict[str, Any],
        headers: dict,
        model: str | None,
        api_key: str | None,
        extra_payload: dict[str, Any] | None = None,
    ) -> RequestParameters:
        """
        Prepare the request to be sent to the provider.

        Each step (api_key, model, headers, url, payload) can be customized in subclasses.
        """
        # api_key from user, or local token, or raise error
        api_key = self._prepare_api_key(api_key)

        # mapped model from HF model ID
        provider_mapping_info = self._prepare_mapping_info(model)

        # default HF headers + user headers (to customize in subclasses)
        headers = self._prepare_headers(headers, api_key)

        # routed URL if HF token, or direct URL (to customize in '_prepare_route' in subclasses)
        url = self._prepare_url(api_key, provider_mapping_info.provider_id)

        # prepare payload (to customize in subclasses)
        payload = self._prepare_payload_as_dict(inputs, parameters, provider_mapping_info=provider_mapping_info)
        if payload is not None:
            payload = recursive_merge(payload, filter_none(extra_payload or {}))

        # body data (to customize in subclasses)
        data = self._prepare_payload_as_bytes(inputs, parameters, provider_mapping_info, extra_payload)

        # check if both payload and data are set and return
        if payload is not None and data is not None:
            raise ValueError("Both payload and data cannot be set in the same request.")
        if payload is None and data is None:
            raise ValueError("Either payload or data must be set in the request.")

        # normalize headers to lowercase and add content-type if not present
        normalized_headers = self._normalize_headers(headers, payload, data)

        return RequestParameters(
            url=url,
            task=self.task,
            model=provider_mapping_info.provider_id,
            json=payload,
            data=data,
            headers=normalized_headers,
        )

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        """
        Return the response in the expected format.

        Override this method in subclasses for customized response handling."""
        return response

    def _prepare_api_key(self, api_key: str | None) -> str:
        """Return the API key to use for the request.

        Usually not overwritten in subclasses."""
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                f"You must provide an api_key to work with {self.provider} API or log in with `hf auth login`."
            )
        return api_key

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        """Return the mapped model ID to use for the request.

        Usually not overwritten in subclasses."""
        if model is None:
            raise ValueError(f"Please provide an HF model ID supported by {self.provider}.")

        # hardcoded mapping for local testing
        if HARDCODED_MODEL_INFERENCE_MAPPING.get(self.provider, {}).get(model):
            return HARDCODED_MODEL_INFERENCE_MAPPING[self.provider][model]

        provider_mapping = None
        for mapping in _fetch_inference_provider_mapping(model):
            if mapping.provider == self.provider:
                provider_mapping = mapping
                break

        if provider_mapping is None:
            raise ValueError(f"Model {model} is not supported by provider {self.provider}.")

        if provider_mapping.task != self.task:
            raise ValueError(
                f"Model {model} is not supported for task {self.task} and provider {self.provider}. "
                f"Supported task: {provider_mapping.task}."
            )

        if provider_mapping.status == "staging":
            logger.warning(
                f"Model {model} is in staging mode for provider {self.provider}. Meant for test purposes only."
            )
        if provider_mapping.status == "error":
            logger.warning(
                f"Our latest automated health check on model '{model}' for provider '{self.provider}' did not complete successfully.  "
                "Inference call might fail."
            )
        return provider_mapping

    def _normalize_headers(
        self, headers: dict[str, Any], payload: dict[str, Any] | None, data: MimeBytes | None
    ) -> dict[str, Any]:
        """Normalize the headers to use for the request.

        Override this method in subclasses for customized headers.
        """
        normalized_headers = {key.lower(): value for key, value in headers.items() if value is not None}
        if normalized_headers.get("content-type") is None:
            if data is not None and data.mime_type is not None:
                normalized_headers["content-type"] = data.mime_type
            elif payload is not None:
                normalized_headers["content-type"] = "application/json"
        return normalized_headers

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        """Return the headers to use for the request.

        Override this method in subclasses for customized headers.
        """
        return {**build_hf_headers(token=api_key), **headers}

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        """Return the URL to use for the request.

        Usually not overwritten in subclasses."""
        base_url = self._prepare_base_url(api_key)
        route = self._prepare_route(mapped_model, api_key)
        return f"{base_url.rstrip('/')}/{route.lstrip('/')}"

    def _prepare_base_url(self, api_key: str) -> str:
        """Return the base URL to use for the request.

        Usually not overwritten in subclasses."""
        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            logger.info(f"Calling '{self.provider}' provider through Hugging Face router.")
            return constants.INFERENCE_PROXY_TEMPLATE.format(provider=self.provider)
        else:
            logger.info(f"Calling '{self.provider}' provider directly.")
            return self.base_url

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        """Return the route to use for the request.

        Override this method in subclasses for customized routes.
        """
        return ""

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        """Return the payload to use for the request, as a dict.

        Override this method in subclasses for customized payloads.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.
        """
        return None

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        """Return the body to use for the request, as bytes.

        Override this method in subclasses for customized body data.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.
        """
        return None


class BaseConversationalTask(TaskProviderHelper):
    """
    Base class for conversational (chat completion) tasks.
    The schema follows the OpenAI API format defined here: https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, provider: str, base_url: str):
        super().__init__(provider=provider, base_url=base_url, task="conversational")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/chat/completions"

    def _prepare_payload_as_dict(
        self,
        inputs: list[dict | ChatCompletionInputMessage],
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        # `model` is serialized first so that a router/proxy can resolve the target provider from a
        # small prefix of the request body instead of buffering the whole payload — `messages` can
        # hold megabytes of base64-encoded images.
        # `parameters` also carries a caller-supplied "model" (see `InferenceClient.chat_completion`),
        # which must not take precedence over the provider-mapped id: drop it from the spread.
        return filter_none(
            {
                "model": provider_mapping_info.provider_id,
                "messages": inputs,
                **{key: value for key, value in parameters.items() if key != "model"},
            }
        )


class AutoRouterConversationalTask(BaseConversationalTask):
    """
    Auto-router for conversational tasks.

    We let the Hugging Face router select the best provider for the model, based on availability and user preferences.
    This is a special case since the selection is done server-side (avoid 1 API call to fetch provider mapping).
    """

    def __init__(self):
        super().__init__(provider="auto", base_url="https://router.huggingface.co")

    def _prepare_base_url(self, api_key: str) -> str:
        """Return the base URL to use for the request.

        Usually not overwritten in subclasses."""
        # Route to the proxy if the api_key is a HF TOKEN
        if not api_key.startswith("hf_"):
            raise ValueError("Cannot select auto-router when using non-Hugging Face API key.")
        else:
            return self.base_url  # No `/auto` suffix in the URL

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        """
        In auto-router, we don't need to fetch provider mapping info.
        We just return a dummy mapping info with provider_id set to the HF model ID.
        """
        if model is None:
            raise ValueError("Please provide an HF model ID.")

        return InferenceProviderMapping(
            provider="auto",
            hf_model_id=model,
            providerId=model,
            status="live",
            task="conversational",
        )


class BaseTextGenerationTask(TaskProviderHelper):
    """
    Base class for text-generation (completion) tasks.
    The schema follows the OpenAI API format defined here: https://platform.openai.com/docs/api-reference/completions
    """

    def __init__(self, provider: str, base_url: str):
        super().__init__(provider=provider, base_url=base_url, task="text-generation")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return filter_none({"prompt": inputs, **parameters, "model": provider_mapping_info.provider_id})


@lru_cache(maxsize=None)
def _fetch_inference_provider_mapping(model: str) -> list["InferenceProviderMapping"]:
    """
    Fetch provider mappings for a model from the Hub.
    """
    from huggingface_hub.hf_api import HfApi

    info = HfApi().model_info(model, expand=["inferenceProviderMapping"])
    provider_mapping = info.inference_provider_mapping
    if provider_mapping is None:
        raise ValueError(f"No provider mapping found for model {model}")
    return provider_mapping


def recursive_merge(dict1: dict, dict2: dict) -> dict:
    return {
        **dict1,
        **{
            key: recursive_merge(dict1[key], value)
            if (key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict))
            else value
            for key, value in dict2.items()
        },
    }
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


from typing import Literal, TypedDict

from typing_extensions import NotRequired


class ReloadRegion(TypedDict):
    startLine: int
    startCol: int
    endLine: int
    endCol: int


class ReloadOperationObject(TypedDict):
    kind: Literal["add", "update", "delete"]
    region: ReloadRegion
    objectType: str
    objectName: str


class ReloadOperationRun(TypedDict):
    kind: Literal["run"]
    region: ReloadRegion
    codeLines: str
    stdout: NotRequired[str]
    stderr: NotRequired[str]


class ReloadOperationException(TypedDict):
    kind: Literal["exception"]
    region: ReloadRegion
    traceback: str


class ReloadOperationError(TypedDict):
    kind: Literal["error"]
    traceback: str


class ReloadOperationUI(TypedDict):
    kind: Literal["ui"]
    updated: bool


class ReloadOperationFile(TypedDict):
    kind: Literal["file"]
    created: bool


class ApiCreateReloadRequest(TypedDict):
    filepath: str
    contents: str
    reloadId: NotRequired[str]


class ApiCreateReloadResponseSuccess(TypedDict):
    status: Literal["created"]
    reloadId: str


class ApiCreateReloadResponseError(TypedDict):
    status: Literal["alreadyReloading", "fileNotFound"]


class ApiCreateReloadResponse(TypedDict):
    res: ApiCreateReloadResponseError | ApiCreateReloadResponseSuccess


class ApiGetReloadRequest(TypedDict):
    reloadId: str


class ApiGetReloadEventSourceData(TypedDict):
    data: (
        ReloadOperationError
        | ReloadOperationException
        | ReloadOperationObject
        | ReloadOperationRun
        | ReloadOperationUI
        | ReloadOperationFile
    )


class ApiGetStatusRequest(TypedDict):
    revision: str


class ApiGetStatusResponse(TypedDict):
    reloading: bool
    uncommited: list[str]


class ApiFetchContentsRequest(TypedDict):
    filepath: str


class ApiFetchContentsResponseError(TypedDict):
    status: Literal["fileNotFound"]


class ApiFetchContentsResponseSuccess(TypedDict):
    status: Literal["ok"]
    contents: str


class ApiFetchContentsResponse(TypedDict):
    res: ApiFetchContentsResponseError | ApiFetchContentsResponseSuccess
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
import time
from collections import deque
from collections.abc import Iterator
from typing import Literal, TypedDict

import httpx2

from ..utils._headers import build_hf_headers
from ..utils._http import hf_raise_for_status
from .sse_client import SSEClient
from .types import ApiGetReloadEventSourceData, ApiGetReloadRequest


HOT_RELOADING_PORT = 7887
CLIENT_TIMEOUT = 20


class MultiReplicaStreamWarning(TypedDict):
    kind: Literal["warning"]
    message: str


class MultiReplicaStreamEvent(TypedDict):
    kind: Literal["event"]
    event: ApiGetReloadEventSourceData


class MultiReplicaStreamReplicaHash(TypedDict):
    kind: Literal["replicaHash"]
    hash: str


class MultiReplicaStreamFullMatch(TypedDict):
    kind: Literal["fullMatch"]


class ReloadClient:
    def __init__(
        self,
        *,
        host: str,
        subdomain: str,
        replica_hash: str,
        token: str | None,
    ):
        base_host = host.replace(subdomain, f"{subdomain}--{HOT_RELOADING_PORT}")
        self.replica_hash = replica_hash
        self.client = httpx2.Client(
            base_url=f"{base_host}/--replicas/+{replica_hash}",
            headers=build_hf_headers(token=token),
            timeout=CLIENT_TIMEOUT,
        )

    def get_reload(self, reload_id: str) -> Iterator[ApiGetReloadEventSourceData] | int:
        req = ApiGetReloadRequest(reloadId=reload_id)
        with self.client.stream("POST", "/get-reload", json=req) as res:
            if res.status_code != 200:
                return res.status_code
            hf_raise_for_status(res)
            for event in SSEClient(res.iter_bytes()).events():
                if event.event == "message":
                    yield json.loads(event.data)
        return None


def multi_replica_reload_events(
    commit_sha: str,
    host: str,
    subdomain: str,
    replica_hashes: list[str],
    token: str | None,
    max_retries: int = 10,
) -> Iterator[
    MultiReplicaStreamWarning | MultiReplicaStreamEvent | MultiReplicaStreamReplicaHash | MultiReplicaStreamFullMatch
]:
    clients = [
        ReloadClient(
            host=host,
            subdomain=subdomain,
            replica_hash=hash,
            token=token,
        )
        for hash in replica_hashes
    ]

    first_client_events: dict[int, ApiGetReloadEventSourceData] = {}
    for client_index, client in enumerate(clients):
        if len(clients) > 1:
            yield {"kind": "replicaHash", "hash": client.replica_hash}

        retries = 0
        while isinstance((events := client.get_reload(commit_sha)), int):
            if (retries := retries + 1) > max_retries:
                raise Exception("Too many retries reached")
            if (status_code := events) not in (200, 204):
                raise Exception(f"Unexpected {status_code=} on `ReloadClient.get_reload`")
            subject = "reloadId" if status_code == 204 else "replica"
            yield {"kind": "warning", "message": f"Retrying on unexpected {subject} not found"}
            time.sleep(2)

        full_match = True
        replay: deque[ApiGetReloadEventSourceData] = deque()
        for event_index, event in enumerate(events):
            if client_index == 0:
                first_client_events[event_index] = event
            elif full_match := full_match and first_client_events.get(event_index) == event:
                replay.append(event)
                continue
            while replay:
                yield {"kind": "event", "event": replay.popleft()}
            yield {"kind": "event", "event": event}

        if client_index > 0 and full_match:
            yield {"kind": "fullMatch"}
<|endoftext|>"""
Vendored file: Server Side Events (SSE) client for Python.

Source:
- Author: Maxime Petazzoni <maxime.petazzoni@bulix.org>
- Repository: https://github.com/mpetazzoni/sseclient
- File: https://github.com/mpetazzoni/sseclient/blob/main/sseclient/__init__.py

License:
- Apache-2.0 (from upstream project)

Provides a generator of SSE received through an existing HTTP response.
"""

import logging

__author__ = 'Maxime Petazzoni <maxime.petazzoni@bulix.org>'
__email__ = 'maxime.petazzoni@bulix.org'
__all__ = ['SSEClient']

_FIELD_SEPARATOR = ':'


class SSEClient:
    """Implementation of a SSE client.

    See http://www.w3.org/TR/2009/WD-eventsource-20091029/ for the
    specification.
    """

    def __init__(self, event_source, char_enc='utf-8'):
        """Initialize the SSE client over an existing, ready to consume
        event source.

        The event source is expected to be a binary stream and have a close()
        method. That would usually be something that implements
        io.BinaryIOBase, like an httplib or urllib3 HTTPResponse object.
        """
        self._logger = logging.getLogger(self.__class__.__module__)
        self._logger.debug('Initialized SSE client from event source %s',
                           event_source)
        self._event_source = event_source
        self._char_enc = char_enc

    def _read(self):
        """Read the incoming event source stream and yield event chunks.

        Unfortunately it is possible for some servers to decide to break an
        event into multiple HTTP chunks in the response. It is thus necessary
        to correctly stitch together consecutive response chunks and find the
        SSE delimiter (empty new line) to yield full, correct event chunks."""
        data = b''
        for chunk in self._event_source:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b'\r\r', b'\n\n', b'\r\n\r\n')):
                    yield data
                    data = b''
        if data:
            yield data

    def events(self):
        for chunk in self._read():
            event = Event()
            # Split before decoding so splitlines() only uses \r and \n
            for line in chunk.splitlines():
                # Decode the line.
                line = line.decode(self._char_enc)

                # Lines starting with a separator are comments and are to be
                # ignored.
                if not line.strip() or line.startswith(_FIELD_SEPARATOR):
                    continue

                data = line.split(_FIELD_SEPARATOR, 1)
                field = data[0]

                # Ignore unknown fields.
                if field not in event.__dict__:
                    self._logger.debug('Saw invalid field %s while parsing '
                                       'Server Side Event', field)
                    continue

                if len(data) > 1:
                    # From the spec:
                    # "If value starts with a single U+0020 SPACE character,
                    # remove it from value."
                    if data[1].startswith(' '):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    # If no value is present after the separator,
                    # assume an empty value.
                    value = ''

                # The data field may come over multiple lines and their values
                # are concatenated with each other.
                if field == 'data':
                    event.__dict__[field] += value + '\n'
                else:
                    event.__dict__[field] = value

            # Events with no data are not dispatched.
            if not event.data:
                continue

            # If the data field ends with a newline, remove it.
            if event.data.endswith('\n'):
                event.data = event.data[0:-1]

            # Empty event names default to 'message'
            event.event = event.event or 'message'

            # Dispatch the event
            self._logger.debug('Dispatching %s...', event)
            yield event

    def close(self):
        """Manually close the event source stream."""
        self._event_source.close()


class Event:
    """Representation of an event from the event stream."""

    def __init__(self, id=None, event='message', data='', retry=None):
        self.id = id
        self.event = event
        self.data = data
        self.retry = retry

    def __str__(self):
        s = f'{self.event} event'
        if self.id:
            s += f' #{self.id}'
        if self.data:
            s += ', {} byte{}'.format(len(self.data),
                                        's' if len(self.data) else '')
        else:
            s += ', no data'
        if self.retry:
            s += f', retry in {self.retry}ms'
        return s
<|endoftext|># Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Contains pytorch-specific helpers."""

import importlib
import importlib.util
import json
import os
import re
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, NamedTuple, Union

from packaging import version

from .. import constants, logging
from ._base import MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory


logger = logging.get_logger(__file__)

SAFETENSORS_EXTENSION = ".safetensors"

# A checkpoint index is plain metadata (a tensor name -> shard file mapping). 10 MB is already far beyond
# anything a real checkpoint produces, so anything bigger is not worth parsing into memory.
MAX_INDEX_FILE_SIZE = 10 * 1024 * 1024

if TYPE_CHECKING:
    import torch

# SAVING


def save_torch_model(
    model: "torch.nn.Module",
    save_directory: str | Path,
    *,
    filename_pattern: str | None = None,
    force_contiguous: bool = True,
    max_shard_size: int | str = MAX_SHARD_SIZE,
    metadata: dict[str, str] | None = None,
    safe_serialization: bool = True,
    is_main_process: bool = True,
    shared_tensors_to_discard: list[str] | None = None,
):
    """
    Saves a given torch model to disk, handling sharding and shared tensors issues.

    See also [`save_torch_state_dict`] to save a state dict with more flexibility.

    For more information about tensor sharing, check out [this guide](https://huggingface.co/docs/safetensors/torch_shared_tensors).

    The model state dictionary is split into shards so that each shard is smaller than a given size. The shards are
    saved in the `save_directory` with the given `filename_pattern`. If the model is too big to fit in a single shard,
    an index file is saved in the `save_directory` to indicate where each tensor is saved. This helper uses
    [`split_torch_state_dict_into_shards`] under the hood. If `safe_serialization` is `True`, the shards are saved as
    safetensors (the default). Otherwise, the shards are saved as pickle.

    Before saving the model, the `save_directory` is cleaned from any previous shard files.

    > [!WARNING]
    > If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    > size greater than `max_shard_size`.

    > [!WARNING]
    > If your model is a `transformers.PreTrainedModel`, you should pass `model._tied_weights_keys` as `shared_tensors_to_discard` to properly handle shared tensors saving. This ensures the correct duplicate tensors are discarded during saving.

    Args:
        model (`torch.nn.Module`):
            The model to save on disk.
        save_directory (`str` or `Path`):
            The directory in which the model will be saved.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"` or `pytorch_model{suffix}.bin` depending on `safe_serialization`
            parameter.
        force_contiguous (`boolean`, *optional*):
            Forcing the state_dict to be saved as contiguous tensors. This has no effect on the correctness of the
            model, but it could potentially change performance if the layout of the tensor was chosen specifically for
            that reason. Defaults to `True`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.
        metadata (`dict[str, str]`, *optional*):
            Extra information to save along with the model. Some metadata will be added for each dropped tensors.
            This information will not be enough to recover the entire shared structure but might help understanding
            things.
        safe_serialization (`bool`, *optional*):
            Whether to save as safetensors, which is the default behavior. If `False`, the shards are saved as pickle.
            Safe serialization is recommended for security reasons. Saving as pickle is deprecated and will be removed
            in a future version.
        is_main_process (`bool`, *optional*):
            Whether the process calling this is the main process or not. Useful when in distributed training like
            TPUs and need to call this function from all processes. In this case, set `is_main_process=True` only on
            the main process to avoid race conditions. Defaults to True.
        shared_tensors_to_discard (`list[str]`, *optional*):
            List of tensor names to drop when saving shared tensors. If not provided and shared tensors are
            detected, it will drop the first name alphabetically.

    Example:

    ```py
    >>> from huggingface_hub import save_torch_model
    >>> model = ... # A PyTorch model

    # Save state dict to "path/to/folder". The model will be split into shards of 5GB each and saved as safetensors.
    >>> save_torch_model(model, "path/to/folder")

    # Load model back
    >>> from huggingface_hub import load_torch_model  # TODO
    >>> load_torch_model(model, "path/to/folder")
    >>>
    ```
    """
    save_torch_state_dict(
        state_dict=model.state_dict(),
        filename_pattern=filename_pattern,
        force_contiguous=force_contiguous,
        max_shard_size=max_shard_size,
        metadata=metadata,
        safe_serialization=safe_serialization,
        save_directory=save_directory,
        is_main_process=is_main_process,
        shared_tensors_to_discard=shared_tensors_to_discard,
    )


def save_torch_state_dict(
    state_dict: dict[str, "torch.Tensor"],
    save_directory: str | Path,
    *,
    filename_pattern: str | None = None,
    force_contiguous: bool = True,
    max_shard_size: int | str = MAX_SHARD_SIZE,
    metadata: dict[str, str] | None = None,
    safe_serialization: bool = True,
    is_main_process: bool = True,
    shared_tensors_to_discard: list[str] | None = None,
) -> None:
    """
    Save a model state dictionary to the disk, handling sharding and shared tensors issues.

    See also [`save_torch_model`] to directly save a PyTorch model.

    For more information about tensor sharing, check out [this guide](https://huggingface.co/docs/safetensors/torch_shared_tensors).

    The model state dictionary is split into shards so that each shard is smaller than a given size. The shards are
    saved in the `save_directory` with the given `filename_pattern`. If the model is too big to fit in a single shard,
    an index file is saved in the `save_directory` to indicate where each tensor is saved. This helper uses
    [`split_torch_state_dict_into_shards`] under the hood. If `safe_serialization` is `True`, the shards are saved as
    safetensors (the default). Otherwise, the shards are saved as pickle.

    Before saving the model, the `save_directory` is cleaned from any previous shard files.

    > [!WARNING]
    > If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    > size greater than `max_shard_size`.

    > [!WARNING]
    > If your model is a `transformers.PreTrainedModel`, you should pass `model._tied_weights_keys` as `shared_tensors_to_discard` to properly handle shared tensors saving. This ensures the correct duplicate tensors are discarded during saving.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dictionary to save.
        save_directory (`str` or `Path`):
            The directory in which the model will be saved.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"` or `pytorch_model{suffix}.bin` depending on `safe_serialization`
            parameter.
        force_contiguous (`boolean`, *optional*):
            Forcing the state_dict to be saved as contiguous tensors. This has no effect on the correctness of the
            model, but it could potentially change performance if the layout of the tensor was chosen specifically for
            that reason. Defaults to `True`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.
        metadata (`dict[str, str]`, *optional*):
            Extra information to save along with the model. Some metadata will be added for each dropped tensors.
            This information will not be enough to recover the entire shared structure but might help understanding
            things.
        safe_serialization (`bool`, *optional*):
            Whether to save as safetensors, which is the default behavior. If `False`, the shards are saved as pickle.
            Safe serialization is recommended for security reasons. Saving as pickle is deprecated and will be removed
            in a future version.
        is_main_process (`bool`, *optional*):
            Whether the process calling this is the main process or not. Useful when in distributed training like
            TPUs and need to call this function from all processes. In this case, set `is_main_process=True` only on
            the main process to avoid race conditions. Defaults to True.
        shared_tensors_to_discard (`list[str]`, *optional*):
            List of tensor names to drop when saving shared tensors. If not provided and shared tensors are
            detected, it will drop the first name alphabetically.

    Example:

    ```py
    >>> from huggingface_hub import save_torch_state_dict
    >>> model = ... # A PyTorch model

    # Save state dict to "path/to/folder". The model will be split into shards of 5GB each and saved as safetensors.
    >>> state_dict = model_to_save.state_dict()
    >>> save_torch_state_dict(state_dict, "path/to/folder")
    ```
    """
    save_directory = str(save_directory)

    if filename_pattern is None:
        filename_pattern = (
            constants.SAFETENSORS_WEIGHTS_FILE_PATTERN
            if safe_serialization
            else constants.PYTORCH_WEIGHTS_FILE_PATTERN
        )

    if metadata is None:
        metadata = {}
    if safe_serialization:
        try:
            from safetensors.torch import save_file as save_file_fn
        except ImportError as e:
            raise ImportError(
                "Please install `safetensors` to use safe serialization. "
                "You can install it with `pip install safetensors`."
            ) from e
        # Clean state dict for safetensors
        state_dict = _clean_state_dict_for_safetensors(
            state_dict,
            metadata,
            force_contiguous=force_contiguous,
            shared_tensors_to_discard=shared_tensors_to_discard,
        )
    else:
        from torch import save as save_file_fn  # type: ignore[assignment, no-redef]

        logger.warning(
            "You are using unsafe serialization. Due to security reasons, it is recommended not to load "
            "pickled models from untrusted sources. If you intend to share your model, we strongly recommend "
            "using safe serialization by installing `safetensors` with `pip install safetensors`."
        )
    # Split dict
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    # Only main process should clean up existing files to avoid race conditions in distributed environment
    if is_main_process:
        # Escape literal parts and use `fullmatch` so that e.g. "model.safetensors.backup" is not deleted.
        prefix, _, suffix = filename_pattern.partition("{suffix}")
        existing_files_regex = re.compile(
            re.escape(prefix) + r"(-\d{5}-of-\d{5})?" + re.escape(suffix) + r"(\.index\.json)?"
        )
        for filename in os.listdir(save_directory):
            if existing_files_regex.fullmatch(filename):
                try:
                    logger.debug(f"Removing existing file '{filename}' from folder.")
                    os.remove(os.path.join(save_directory, filename))
                except Exception as e:
                    logger.warning(
                        f"Error when trying to remove existing '{filename}' from folder: {e}. Continuing..."
                    )

    # Save each shard
    per_file_metadata = {"format": "pt"}
    if not state_dict_split.is_sharded:
        per_file_metadata.update(metadata)
    safe_file_kwargs = {"metadata": per_file_metadata} if safe_serialization else {}
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        save_file_fn(shard, os.path.join(save_directory, filename), **safe_file_kwargs)  # ty: ignore[invalid-argument-type]
        logger.debug(f"Shard saved to {filename}")

    # Save the index (if any)
    if state_dict_split.is_sharded:
        index_path = filename_pattern.format(suffix="") + ".index.json"
        index = {
            "metadata": {**state_dict_split.metadata, **metadata},
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, index_path), "w") as f:
            json.dump(index, f, indent=2)
        logger.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}). "
            f"Model weighs have been saved in {len(state_dict_split.filename_to_tensors)} checkpoint shards. "
            f"You can find where each parameters has been saved in the index located at {index_path}."
        )

    logger.info(f"Model weights successfully saved to {save_directory}!")


def split_torch_state_dict_into_shards(
    state_dict: dict[str, "torch.Tensor"],
    *,
    filename_pattern: str = constants.SAFETENSORS_WEIGHTS_FILE_PATTERN,
    max_shard_size: int | str = MAX_SHARD_SIZE,
) -> StateDictSplit:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].


    > [!TIP]
    > To save a model state dictionary to the disk, see [`save_torch_state_dict`]. This helper uses
    > `split_torch_state_dict_into_shards` under the hood.

    > [!WARNING]
    > If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    > size greater than `max_shard_size`.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dictionary to save.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.

    Example:
    ```py
    >>> import json
    >>> import os
    >>> from safetensors.torch import save_file as safe_save_file
    >>> from huggingface_hub import split_torch_state_dict_into_shards

    >>> def save_state_dict(state_dict: dict[str, torch.Tensor], save_directory: str):
    ...     state_dict_split = split_torch_state_dict_into_shards(state_dict)
    ...     for filename, tensors in state_dict_split.filename_to_tensors.items():
    ...         shard = {tensor: state_dict[tensor] for tensor in tensors}
    ...         safe_save_file(
    ...             shard,
    ...             os.path.join(save_directory, filename),
    ...             metadata={"format": "pt"},
    ...         )
    ...     if state_dict_split.is_sharded:
    ...         index = {
    ...             "metadata": state_dict_split.metadata,
    ...             "weight_map": state_dict_split.tensor_to_filename,
    ...         }
    ...         with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
    ...             f.write(json.dumps(index, indent=2))
    ```
    """
    return split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern=filename_pattern,
        get_storage_size=get_torch_storage_size,
        get_storage_id=get_torch_storage_id,
    )


# LOADING


def load_torch_model(
    model: "torch.nn.Module",
    checkpoint_path: str | os.PathLike,
    *,
    strict: bool = False,
    safe: bool = True,
    weights_only: bool = True,
    map_location: Union[str, "torch.device"] | None = None,
    mmap: bool = False,
    filename_pattern: str | None = None,
) -> NamedTuple:
    """
    Load a checkpoint into a model, handling both sharded and non-sharded checkpoints.

    Args:
        model (`torch.nn.Module`):
            The model in which to load the checkpoint.
        checkpoint_path (`str` or `os.PathLike`):
            Path to either the checkpoint file or directory containing the checkpoint(s).
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the checkpoint.
            As with [`torch.nn.Module.load_state_dict`], the check happens once the checkpoint has been loaded, so
            the model may have been partially updated when the error is raised.
        safe (`bool`, *optional*, defaults to `True`):
            If `safe` is True, the safetensors files will be loaded. If `safe` is False, the function
            will first attempt to load safetensors files if they are available, otherwise it will fall back to loading
            pickle files. A `filename_pattern` that does not describe safetensors files is rejected when `safe=True`.
        weights_only (`bool`, *optional*, defaults to `True`):
            If True, only loads the model weights without optimizer states and other metadata, using torch's
            restricted unpickler. Set to False to allow arbitrary Python objects in a pickle checkpoint (this
            executes arbitrary code at load time). Has no effect on PyTorch < 1.13, which has no restricted
            unpickler and always unpickles without restriction.
        map_location (`str` or `torch.device`, *optional*):
            A `torch.device` object, string or a dict specifying how to remap storage locations. It
            indicates the location where all tensors should be loaded.
        mmap (`bool`, *optional*, defaults to `False`):
            Whether to use memory-mapped file loading. Memory mapping can improve loading performance
            for large models in PyTorch >= 2.1.0 with zipfile-based checkpoints.
        filename_pattern (`str`, *optional*):
            The pattern to look for the index file. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields.
            - `missing_keys` is a list of str containing the missing keys, i.e. keys that are in the model but not in the checkpoint.
            - `unexpected_keys` is a list of str containing the unexpected keys, i.e. keys that are in the checkpoint but not in the model.

    Raises:
        [`FileNotFoundError`](https://docs.python.org/3/library/exceptions.html#FileNotFoundError)
            If the checkpoint file or directory does not exist.
        [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
            If safetensors or torch is not installed when trying to load a .safetensors file or a PyTorch checkpoint respectively.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the checkpoint format cannot be determined.

    Example:
    ```python
    >>> from huggingface_hub import load_torch_model
    >>> model = ... # A PyTorch model
    >>> load_torch_model(model, "path/to/checkpoint")
    ```
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
    # 1. Check if checkpoint is a single file
    if checkpoint_path.is_file():
        state_dict = load_state_dict_from_file(
            checkpoint_file=checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
            mmap=mmap,
            safe=safe,
        )
        return model.load_state_dict(state_dict, strict=strict)

    # 2. If not, checkpoint_path is a directory
    if filename_pattern is None:
        filename_pattern = constants.SAFETENSORS_WEIGHTS_FILE_PATTERN
        index_path = checkpoint_path / (filename_pattern.format(suffix="") + ".index.json")
        # Only fallback to pickle format if safetensors index is not found and safe is False.
        if not index_path.is_file() and not safe:
            filename_pattern = constants.PYTORCH_WEIGHTS_FILE_PATTERN
    elif safe and not _is_safetensors(filename_pattern.format(suffix="")):
        # A pickle `filename_pattern` combined with `safe=True` is a conflict: loading those shards would execute
        # arbitrary code at load time. Refuse it instead of silently letting the pattern win over `safe`.
        raise ValueError(
            f"`filename_pattern={filename_pattern!r}` does not describe safetensors files but `safe=True`. "
            "Pass `safe=False` to explicitly allow loading pickled weights (this executes arbitrary code at load time)."
        )

    index_path = checkpoint_path / (filename_pattern.format(suffix="") + ".index.json")

    if index_path.is_file():
        return _load_sharded_checkpoint(
            model=model,
            save_directory=checkpoint_path,
            strict=strict,
            weights_only=weights_only,
            safe=safe,
            filename_pattern=filename_pattern,
        )

    # Look for single model file.
    # Safetensors is tried first even when `safe=False`: the docstring promises a *fallback* to pickle files for
    # that case, not a pickle-first preference. A directory holding only `model.safetensors` used to raise
    # `ValueError` under `safe=False`.
    model_files = list(checkpoint_path.glob("*" + SAFETENSORS_EXTENSION))
    if len(model_files) != 1 and not safe:
        # Fallback only helps if it yields exactly one file, so it must also run when several safetensors files
        # were found (e.g. `model.safetensors` next to `model.fp16.safetensors` and a single `pytorch_model.bin`).
        model_files = list(checkpoint_path.glob("*.bin"))
    if len(model_files) == 1:
        state_dict = load_state_dict_from_file(
            checkpoint_file=model_files[0],
            map_location=map_location,
            weights_only=weights_only,
            mmap=mmap,
            safe=safe,
        )
        return model.load_state_dict(state_dict, strict=strict)

    raise ValueError(
        f"Directory '{checkpoint_path}' does not contain a valid checkpoint. "
        "Expected either a sharded checkpoint with an index file, or a single model file."
    )


def _load_sharded_checkpoint(
    model: "torch.nn.Module",
    save_directory: os.PathLike,
    *,
    strict: bool = False,
    weights_only: bool = True,
    safe: bool = True,
    filename_pattern: str = constants.SAFETENSORS_WEIGHTS_FILE_PATTERN,
) -> NamedTuple:
    """
    Loads a sharded checkpoint into a model. This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint. Each shard is loaded one by one and removed from memory after being loaded into the model.

    Args:
        model (`torch.nn.Module`):
            The model in which to load the checkpoint.
        save_directory (`str` or `os.PathLike`):
            A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
            As with [`torch.nn.Module.load_state_dict`], the check happens once every shard has been loaded, so the
            model may have been partially updated when the error is raised.
        weights_only (`bool`, *optional*, defaults to `True`):
            If True, only loads the model weights without optimizer states and other metadata, using torch's
            restricted unpickler. Set to False to allow arbitrary Python objects in a pickle checkpoint (this
            executes arbitrary code at load time). Has no effect on PyTorch < 1.13, which has no restricted
            unpickler and always unpickles without restriction.
        safe (`bool`, *optional*, defaults to `True`):
            If True, every shard is loaded with the safetensors loader. If False, shards are loaded as safetensors
            when their name says so and with `torch.load` otherwise.
        filename_pattern (`str`, *optional*, defaults to `"model{suffix}.safetensors"`):
            The pattern to look for the index file. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields,
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """

    # 1. Load and validate index file
    # The index file contains mapping of parameter names to shard files
    index_path = filename_pattern.format(suffix="") + ".index.json"
    index_file = os.path.join(save_directory, index_path)
    # Refuse oversized index files before parsing them: the index is metadata, never a multi-GB payload.
    if os.path.getsize(index_file) > MAX_INDEX_FILE_SIZE:
        raise ValueError(
            f"Invalid index file '{index_file}': larger than {MAX_INDEX_FILE_SIZE} bytes, which is not a valid "
            "checkpoint index."
        )
    with open(index_file, encoding="utf-8") as f:
        index = json.load(f)
    # Validate the structure before touching it: a malformed index would otherwise surface as a raw `KeyError` /
    # `AttributeError` / `TypeError` traceback from attacker-controlled input.
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"Invalid index file '{index_file}': expected a JSON object with a 'weight_map' object.")
    if not all(isinstance(shard_file, str) for shard_file in index["weight_map"].values()):
        raise ValueError(f"Invalid index file '{index_file}': all 'weight_map' values must be strings.")

    # 2. Validate shard filenames from the index
    # This prevents path traversal attacks and extension confusion attacks
    # (e.g. a safetensors index referencing .bin pickle files)
    expected_extension = Path(filename_pattern.format(suffix="")).suffix  # e.g. ".safetensors"
    shard_files = list(set(index["weight_map"].values()))
    for shard_file in shard_files:
        # Reject anything that could escape `save_directory` on any host OS:
        # POSIX absolute ("/tmp/x"), Windows drive ("C:x", "C:\\x"), UNC
        # ("\\\\server\\share\\x"), rooted-without-drive ("\\x", "/x"), or
        # ".." traversal — including "..\\x" which `os.path.isabs` never caught on POSIX.
        #
        # We parse with `PureWindowsPath` *regardless of host OS*: it treats both "/" and
        # "\\" as separators and exposes `drive` / `root`, so a single check rejects a
        # malicious index file on Linux too (e.g. if it's later opened on Windows). The
        # only over-strict case is a POSIX filename like "a:foo" which would be parsed as
        # drive "a:" — such names are never produced for safetensors shards and would
        # break on Windows anyway, so rejecting them is fine.
        win_path = PureWindowsPath(shard_file)
        if win_path.drive or win_path.root or ".." in win_path.parts:
            raise ValueError(
                f"Invalid shard filename '{shard_file}' in index file '{index_file}'. "
                "Shard filenames must be relative paths without '..' components."
            )
        # Reject extension mismatch (e.g. .bin shard in a .safetensors index). Note this check is deliberately
        # case-*sensitive* while `_is_safetensors` (the loader routing hint) is not: a legitimate index produced by
        # `save_torch_state_dict` is always lowercase, and there is no reason to accept anything else from an index
        # file. An index entry spelled `.safetensors` stays routed to the safetensors loader even on a
        # case-insensitive filesystem resolving it to an uppercase file on disk, so it fails instead of executing.
        if not shard_file.endswith(expected_extension):
            raise ValueError(
                f"Invalid shard filename '{shard_file}' in index file '{index_file}'. "
                f"Expected '{expected_extension}' extension to match the index format."
            )

    # 3. Load each shard using `load_state_dict`
    # Get unique shard files (multiple parameters can be in same shard)
    loaded_keys: set[str] = set()
    for shard_file in shard_files:
        # Load shard into memory
        shard_path = os.path.join(save_directory, shard_file)
        state_dict = load_state_dict_from_file(
            shard_path,
            map_location="cpu",
            weights_only=weights_only,
            safe=safe,
        )
        loaded_keys.update(state_dict.keys())
        # Update model with parameters from this shard. `strict=False` here: a single shard never holds all the
        # model's keys, so per-shard strict loading would always raise. Strictness is enforced against the real
        # loaded keys below.
        model.load_state_dict(state_dict, strict=False)
        # Explicitly remove the state dict from memory
        del state_dict

    # 4. Validate keys and return compatibility info.
    # Both are computed from what the shards actually contained, never from the index file: the index is
    # attacker-controlled metadata and must not be able to lie about what was loaded into the model — neither by
    # hiding a tensor it did declare, nor by failing `strict=True` over a tensor no shard ever held.
    if unexpected := loaded_keys - set(index["weight_map"]):
        logger.warning(f"Shard files contain tensors absent from the index: {sorted(unexpected)}")
    if strict:
        _validate_keys_for_strict_loading(model, loaded_keys)
    model_keys = set(model.state_dict().keys())
    return _IncompatibleKeys(
        missing_keys=list(model_keys - loaded_keys), unexpected_keys=list(loaded_keys - model_keys)
    )


def load_state_dict_from_file(
    checkpoint_file: str | os.PathLike,
    map_location: Union[str, "torch.device"] | None = None,
    weights_only: bool = True,
    mmap: bool = False,
    *,
    safe: bool = True,
) -> dict[str, "torch.Tensor"] | Any:
    """
    Loads a checkpoint file, handling both safetensors and pickle checkpoint formats.

    Args:
        checkpoint_file (`str` or `os.PathLike`):
            Path to the checkpoint file to load. Can be either a safetensors or pickle (`.bin`) checkpoint.
        map_location (`str` or `torch.device`, *optional*):
            A `torch.device` object, string or a dict specifying how to remap storage locations. It
            indicates the location where all tensors should be loaded.
        weights_only (`bool`, *optional*, defaults to `True`):
            If True, only loads the model weights without optimizer states and other metadata, using torch's
            restricted unpickler. Set to False to allow arbitrary Python objects in a pickle checkpoint (this
            executes arbitrary code at load time). Has no effect when loading safetensors files, nor on
            PyTorch < 1.13 which has no restricted unpickler — those versions always unpickle without restriction
            and a warning is logged.
        mmap (`bool`, *optional*, defaults to `False`):
            Whether to use memory-mapped file loading. Memory mapping can improve loading performance
            for large models in PyTorch >= 2.1.0 with zipfile-based checkpoints. Has no effect when
            loading safetensors files, as the `safetensors` library uses memory mapping by default.
        safe (`bool`, *optional*, defaults to `True`):
            If True, the checkpoint is always loaded as safetensors, whatever its name. Any other format
            (e.g. a pickle `.bin` file) raises a `ValueError` instead of being deserialized: pickle checkpoints can
            execute arbitrary code at load time. If False, the file is loaded as safetensors when its name says so,
            and with `torch.load` otherwise.

    Returns:
        `Union[dict[str, "torch.Tensor"], Any]`: The loaded checkpoint.
            - For safetensors files: always returns a dictionary mapping parameter names to tensors.
            - For pickle files: returns any Python object that was pickled (commonly a state dict, but could be
              an entire model, optimizer state, or any other Python object).

    Raises:
        [`FileNotFoundError`](https://docs.python.org/3/library/exceptions.html#FileNotFoundError)
            If the checkpoint file does not exist.
        [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
            If safetensors or torch is not installed when trying to load a .safetensors file or a PyTorch checkpoint respectively.
        [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
            If the checkpoint file format is invalid or if git-lfs files are not properly downloaded.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the checkpoint file path is empty or invalid, or if it cannot be deserialized as safetensors while
            `safe=True` (pass `safe=False` to allow pickle checkpoints).

    Example:
    ```python
    >>> from huggingface_hub import load_state_dict_from_file

    # Load a safetensors checkpoint (safe by default)
    >>> state_dict = load_state_dict_from_file("path/to/model.safetensors", safe=True)
    >>> model.load_state_dict(state_dict)

    # Load a pickle checkpoint. `safe=False` is required: pickle files can execute arbitrary code.
    >>> state_dict = load_state_dict_from_file("path/to/model.bin", safe=False, map_location="cpu")
    >>> model.load_state_dict(state_dict)
    ```
    """
    checkpoint_path = Path(checkpoint_file)

    # Check if file exists and is a regular file (not a directory)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint file found at '{checkpoint_path}'. Please verify the path is correct and "
            "the file has been properly downloaded."
        )

    # Load safetensors checkpoint
    if safe or _is_safetensors(checkpoint_path):
        # `safe=True` is a hard guarantee: the safetensors loader is used whatever the file is called. A pickle file
        # pretending to be safetensors — or any other name the extension check failed to recognize — fails to
        # deserialize instead of being executed. The filename is only a hint, used in the already-unsafe `safe=False`
        # branch below.
        try:
            return _load_safetensors_file(checkpoint_path, map_location=map_location)
        except ImportError:
            raise
        except OSError:
            raise  # invalid safetensors metadata: the error message is already actionable
        except Exception as e:
            if safe:
                raise ValueError(
                    f"Cannot load '{checkpoint_path}' as safetensors. If this is a pickle checkpoint, pass "
                    "`safe=False` to allow it (this executes arbitrary code at load time)."
                ) from e
            raise

    # Otherwise, load from pickle
    try:
        import torch
        from torch import load
    except ImportError as e:
        raise ImportError(
            "Please install `torch` to load torch tensors. You can install it with `pip install torch`."
        ) from e

    # Add additional kwargs, mmap is only supported in torch >= 2.1.0
    additional_kwargs = {}
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        additional_kwargs["mmap"] = mmap

    # weights_only is only supported in torch >= 1.13.0
    if version.parse(torch.__version__) >= version.parse("1.13.0"):
        additional_kwargs["weights_only"] = weights_only

    if not additional_kwargs.get("weights_only", False):
        # Warn only when the unrestricted unpickler is what actually runs: either the caller asked for it, or torch
        # is too old to have a restricted one and `weights_only=True` could not be honored. With the restricted
        # unpickler in play the warning would be both untrue and noisy (it is emitted once per shard).
        logger.warning(
            f"Loading '{checkpoint_path}' with `torch.load` and no restricted unpickler (`weights_only=False`, or "
            "torch < 1.13 which does not support it). Pickle checkpoints can execute arbitrary code at load time; "
            "only load files from sources you trust."
        )

    return load(
        checkpoint_file,
        map_location=map_location,
        **additional_kwargs,
    )


def _load_safetensors_file(
    checkpoint_file: str | os.PathLike,
    map_location: Union[str, "torch.device"] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Load a safetensors checkpoint. Raises `safetensors.SafetensorError` if the file is not a valid safetensors."""
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "Please install `safetensors` to load safetensors checkpoint. "
            "You can install it with `pip install safetensors`."
        ) from e

    # Check format of the archive
    with safe_open(checkpoint_file, framework="pt") as f:  # type: ignore[attr-defined]
        metadata = f.metadata()
    # see comment: https://github.com/huggingface/transformers/blob/3d213b57fe74302e5902d68ed9478c3ad1aaa713/src/transformers/modeling_utils.py#L3966
    if metadata is not None and metadata.get("format") not in ["pt", "mlx"]:
        raise OSError(
            f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
            "you save your model with the `save_torch_model` method."
        )
    if map_location is None:
        device = "cpu"  # `load_file`'s own default, spelled out so `device` is always a `str`
    else:
        device = str(map_location.type) if hasattr(map_location, "type") else str(map_location)
    # meta device is not supported with safetensors, falling back to CPU
    if device == "meta":
        logger.warning("Meta device is not supported with safetensors. Falling back to CPU device.")
        device = "cpu"
    return load_file(checkpoint_file, device=device)


# HELPERS


def _is_safetensors(filename: Union[str, os.PathLike]) -> bool:
    """Whether `filename` must be loaded with the safetensors loader.

    The comparison is case-insensitive on purpose: Windows and macOS resolve filenames case-insensitively, so a file
    named `model.SAFETENSORS` can be picked up by a case-insensitive glob. This is only a *hint* used when `safe=False`;
    it is never the security boundary (see `load_state_dict_from_file`).
    """
    return str(filename).lower().endswith(SAFETENSORS_EXTENSION)


def _validate_keys_for_strict_loading(
    model: "torch.nn.Module",
    loaded_keys: Iterable[str],
) -> None:
    """
    Validate that model keys match loaded keys when strict loading is enabled.

    Args:
        model: The PyTorch model being loaded
        loaded_keys: The keys present in the checkpoint

    Raises:
        RuntimeError: If there are missing or unexpected keys in strict mode
    """
    loaded_keys_set = set(loaded_keys)
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - loaded_keys_set  # Keys in model but not in checkpoint
    unexpected_keys = loaded_keys_set - model_keys  # Keys in checkpoint but not in model

    if missing_keys or unexpected_keys:
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if missing_keys:
            str_missing_keys = ",".join([f'"{k}"' for k in sorted(missing_keys)])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if unexpected_keys:
            str_unexpected_keys = ",".join([f'"{k}"' for k in sorted(unexpected_keys)])
            error_message += f"\nUnexpected key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)


def _get_unique_id(tensor: "torch.Tensor") -> int | tuple[Any, ...]:
    """Returns a unique id for plain tensor
    or a (potentially nested) Tuple of unique id for the flattened Tensor
    if the input is a wrapper tensor subclass Tensor
    """

    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            local_tensor = tensor.to_local()
            return local_tensor.storage().data_ptr()
    except ImportError:
        pass

    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            attrs, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            return tuple(_get_unique_id(getattr(tensor, attr)) for attr in attrs)

    except ImportError:
        # for torch version less than 2.1, we can fall back to original implementation
        pass

    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors don't have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla  # type: ignore[import]

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return unique_id


def get_torch_storage_id(tensor: "torch.Tensor") -> tuple["torch.device", int | tuple[Any, ...], int] | None:
    """
    Return unique identifier to a tensor storage.

    Multiple different tensors can share the same underlying storage. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    In the case of meta tensors, we return None since we can't tell if they share the same storage.

    Taken from https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/pytorch_utils.py#L278.
    """
    if tensor.device.type == "meta":
        return None
    else:
        return tensor.device, _get_unique_id(tensor), get_torch_storage_size(tensor)


def get_torch_storage_size(tensor: "torch.Tensor") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L31C1-L41C59
    """
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            # this returns the size of the FULL tensor in bytes
            return tensor.nbytes
    except ImportError:
        pass

    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            attrs, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            return sum(get_torch_storage_size(getattr(tensor, attr)) for attr in attrs)
    except ImportError:
        # for torch version less than 2.1, we can fall back to original implementation
        pass

    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _get_dtype_size(tensor.dtype)
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _get_dtype_size(tensor.dtype)


@lru_cache
def is_torch_tpu_available(check_device=True):
    """
    Checks if `torch_xla` is installed and potentially if a TPU is in the environment

    Taken from https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/utils/import_utils.py#L463.
    """
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm  # type: ignore[import]

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


def storage_ptr(tensor: "torch.Tensor") -> int | tuple[Any, ...]:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L11.
    """
    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            return _get_unique_id(tensor)  # type: ignore
    except ImportError:
        # for torch version less than 2.1, we can fall back to original implementation
        pass

    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _clean_state_dict_for_safetensors(
    state_dict: dict[str, "torch.Tensor"],
    metadata: dict[str, str],
    force_contiguous: bool = True,
    shared_tensors_to_discard: list[str] | None = None,
):
    """Remove shared tensors from state_dict and update metadata accordingly (for reloading).

    Warning: `state_dict` and `metadata` are mutated in-place!

    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L155.
    """
    to_removes = _remove_duplicate_names(state_dict, discard_names=shared_tensors_to_discard)
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if metadata is None:
                metadata = {}

            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            del state_dict[to_remove]
    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    return state_dict


def _end_ptr(tensor: "torch.Tensor") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L23.
    """
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _get_dtype_size(tensor.dtype)
    else:
        stop = tensor.data_ptr()
    return stop


def _filter_shared_not_shared(tensors: list[set[str]], state_dict: dict[str, "torch.Tensor"]) -> list[set[str]]:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L44
    """
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors


def _find_shared_tensors(state_dict: dict[str, "torch.Tensor"]) -> list[set[str]]:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L69.
    """
    import torch

    tensors_dict = defaultdict(set)
    for k, v in state_dict.items():
        if v.device != torch.device("meta") and storage_ptr(v) != 0 and get_torch_storage_size(v) != 0:
            # Need to add device as key because of multiple GPU.
            tensors_dict[(v.device, storage_ptr(v), get_torch_storage_size(v))].add(k)
    tensors = list(sorted(tensors_dict.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors


def _is_complete(tensor: "torch.Tensor") -> bool:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L80
    """
    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            attrs, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            return all(_is_complete(getattr(tensor, attr)) for attr in attrs)
    except ImportError:
        # for torch version less than 2.1, we can fall back to original implementation
        pass

    return tensor.data_ptr() == storage_ptr(tensor) and tensor.nelement() * _get_dtype_size(
        tensor.dtype
    ) == get_torch_storage_size(tensor)


def _remove_duplicate_names(
    state_dict: dict[str, "torch.Tensor"],
    *,
    preferred_names: list[str] | None = None,
    discard_names: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Taken from https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L80
    """
    if preferred_names is None:
        preferred_names = []
    unique_preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    unique_discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = {name for name in shared if _is_complete(state_dict[name])}
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage. Refusing to save/load the model"
                " since you could be storing much more memory than needed. Please refer to"
                " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                " issue."
            )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(unique_discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if unique_preferred_names:
            preferred = unique_preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


@lru_cache
def _get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/b7c0f38b6ae072c3cc6208933df0c81fbd2ef837/bindings/python/py_src/safetensors/torch.py#L398
    """
    import torch

    # torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e4m3fnuz = getattr(torch, "float8_e4m3fnuz", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _float8_e5m2fnuz = getattr(torch, "float8_e5m2fnuz", None)
    _float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    _float4_e2m1_x2 = getattr(torch, "float4_e2m1fn_x2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        torch.complex64: 8,
        _float8_e4m3fn: 1,
        _float8_e4m3fnuz: 1,
        _float8_e5m2: 1,
        _float8_e5m2fnuz: 1,
        _float8_e8m0: 1,
        _float4_e2m1_x2: 1,
    }

    if hasattr(torch, "uint64"):  # Torch 2.3.0+
        _SIZE.update(
            {
                torch.uint64: 8,
                torch.uint32: 4,
                torch.uint16: 2,
            }
        )

    return _SIZE[dtype]


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    """
    This is used to report missing and unexpected keys in the state dict.
    Taken from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L52.

    """

    def __repr__(self) -> str:
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__
<|endoftext|>import json
import logging
import mmap
import os
import shutil
import zipfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DDUFCorruptedFileError, DDUFExportError, DDUFInvalidEntryNameError


logger = logging.getLogger(__name__)

DDUF_ALLOWED_ENTRIES = {
    # Allowed file extensions in a DDUF file
    ".json",
    ".model",
    ".safetensors",
    ".txt",
}

DDUF_FOLDER_REQUIRED_ENTRIES = {
    # Each folder must contain at least one of these entries
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "scheduler_config.json",
}


@dataclass
class DDUFEntry:
    """Object representing a file entry in a DDUF file.

    See [`read_dduf_file`] for how to read a DDUF file.

    Attributes:
        filename (str):
            The name of the file in the DDUF archive.
        offset (int):
            The offset of the file in the DDUF archive.
        length (int):
            The length of the file in the DDUF archive.
        dduf_path (str):
            The path to the DDUF archive (for internal use).
    """

    filename: str
    length: int
    offset: int

    dduf_path: Path = field(repr=False)

    @contextmanager
    def as_mmap(self) -> Generator[bytes, None, None]:
        """Open the file as a memory-mapped file.

        Useful to load safetensors directly from the file.

        Example:
            ```py
            >>> import safetensors.torch
            >>> with entry.as_mmap() as mm:
            ...     tensors = safetensors.torch.load(mm)
            ```
        """
        with self.dduf_path.open("rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                yield mm[self.offset : self.offset + self.length]

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the file as text.

        Useful for '.txt' and '.json' entries.

        Example:
            ```py
            >>> import json
            >>> index = json.loads(entry.read_text())
            ```
        """
        with self.dduf_path.open("rb") as f:
            f.seek(self.offset)
            return f.read(self.length).decode(encoding=encoding)


def read_dduf_file(dduf_path: os.PathLike | str) -> dict[str, DDUFEntry]:
    """
    Read a DDUF file and return a dictionary of entries.

    Only the metadata is read, the data is not loaded in memory.

    Args:
        dduf_path (`str` or `os.PathLike`):
            The path to the DDUF file to read.

    Returns:
        `dict[str, DDUFEntry]`:
            A dictionary of [`DDUFEntry`] indexed by filename.

    Raises:
        - [`DDUFCorruptedFileError`]: If the DDUF file is corrupted (i.e. doesn't follow the DDUF format).

    Example:
        ```python
        >>> import json
        >>> import safetensors.torch
        >>> from huggingface_hub import read_dduf_file

        # Read DDUF metadata
        >>> dduf_entries = read_dduf_file("FLUX.1-dev.dduf")

        # Returns a mapping filename <> DDUFEntry
        >>> dduf_entries["model_index.json"]
        DDUFEntry(filename='model_index.json', offset=66, length=587)

        # Load model index as JSON
        >>> json.loads(dduf_entries["model_index.json"].read_text())
        {'_class_name': 'FluxPipeline', '_diffusers_version': '0.32.0.dev0', '_name_or_path': 'black-forest-labs/FLUX.1-dev', ...

        # Load VAE weights using safetensors
        >>> with dduf_entries["vae/diffusion_pytorch_model.safetensors"].as_mmap() as mm:
        ...     state_dict = safetensors.torch.load(mm)
        ```
    """
    entries = {}
    dduf_path = Path(dduf_path)
    logger.info(f"Reading DDUF file {dduf_path}")
    with zipfile.ZipFile(str(dduf_path), "r") as zf:
        for info in zf.infolist():
            logger.debug(f"Reading entry {info.filename}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise DDUFCorruptedFileError("Data must not be compressed in DDUF file.")

            try:
                _validate_dduf_entry_name(info.filename)
            except DDUFInvalidEntryNameError as e:
                raise DDUFCorruptedFileError(f"Invalid entry name in DDUF file: {info.filename}") from e

            offset = _get_data_offset(zf, info)

            entries[info.filename] = DDUFEntry(
                filename=info.filename, offset=offset, length=info.file_size, dduf_path=dduf_path
            )

    # Consistency checks on the DDUF file
    if "model_index.json" not in entries:
        raise DDUFCorruptedFileError("Missing required 'model_index.json' entry in DDUF file.")
    index = json.loads(entries["model_index.json"].read_text())
    _validate_dduf_structure(index, entries.keys())

    logger.info(f"Done reading DDUF file {dduf_path}. Found {len(entries)} entries")
    return entries


def export_entries_as_dduf(dduf_path: str | os.PathLike, entries: Iterable[tuple[str, str | Path | bytes]]) -> None:
    """Write a DDUF file from an iterable of entries.

    This is a lower-level helper than [`export_folder_as_dduf`] that allows more flexibility when serializing data.
    In particular, you don't need to save the data on disk before exporting it in the DDUF file.

    Args:
        dduf_path (`str` or `os.PathLike`):
            The path to the DDUF file to write.
        entries (`Iterable[tuple[str, Union[str, Path, bytes]]]`):
            An iterable of entries to write in the DDUF file. Each entry is a tuple with the filename and the content.
            The filename should be the path to the file in the DDUF archive.
            The content can be a string or a pathlib.Path representing a path to a file on the local disk or directly the content as bytes.

    Raises:
        - [`DDUFExportError`]: If anything goes wrong during the export (e.g. invalid entry name, missing 'model_index.json', etc.).

    Example:
        ```python
        # Export specific files from the local disk.
        >>> from huggingface_hub import export_entries_as_dduf
        >>> export_entries_as_dduf(
        ...     dduf_path="stable-diffusion-v1-4-FP16.dduf",
        ...     entries=[ # List entries to add to the DDUF file (here, only FP16 weights)
        ...         ("model_index.json", "path/to/model_index.json"),
        ...         ("vae/config.json", "path/to/vae/config.json"),
        ...         ("vae/diffusion_pytorch_model.fp16.safetensors", "path/to/vae/diffusion_pytorch_model.fp16.safetensors"),
        ...         ("text_encoder/config.json", "path/to/text_encoder/config.json"),
        ...         ("text_encoder/model.fp16.safetensors", "path/to/text_encoder/model.fp16.safetensors"),
        ...         # ... add more entries here
        ...     ]
        ... )
        ```

        ```python
        # Export state_dicts one by one from a loaded pipeline
        >>> from diffusers import DiffusionPipeline
        >>> from typing import Generator, Tuple
        >>> import safetensors.torch
        >>> from huggingface_hub import export_entries_as_dduf
        >>> pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        ... # ... do some work with the pipeline

        >>> def as_entries(pipe: DiffusionPipeline) -> Generator[tuple[str, bytes], None, None]:
        ...     # Build a generator that yields the entries to add to the DDUF file.
        ...     # The first element of the tuple is the filename in the DDUF archive (must use UNIX separator!). The second element is the content of the file.
        ...     # Entries will be evaluated lazily when the DDUF file is created (only 1 entry is loaded in memory at a time)
        ...     yield "vae/config.json", pipe.vae.to_json_string().encode()
        ...     yield "vae/diffusion_pytorch_model.safetensors", safetensors.torch.save(pipe.vae.state_dict())
        ...     yield "text_encoder/config.json", pipe.text_encoder.config.to_json_string().encode()
        ...     yield "text_encoder/model.safetensors", safetensors.torch.save(pipe.text_encoder.state_dict())
        ...     # ... add more entries here

        >>> export_entries_as_dduf(dduf_path="stable-diffusion-v1-4.dduf", entries=as_entries(pipe))
        ```
    """
    logger.info(f"Exporting DDUF file '{dduf_path}'")
    filenames = set()
    index = None
    with zipfile.ZipFile(str(dduf_path), "w", zipfile.ZIP_STORED) as archive:
        for filename, content in entries:
            if filename in filenames:
                raise DDUFExportError(f"Can't add duplicate entry: {filename}")
            filenames.add(filename)

            if filename == "model_index.json":
                try:
                    index = json.loads(_load_content(content).decode())
                except json.JSONDecodeError as e:
                    raise DDUFExportError("Failed to parse 'model_index.json'.") from e

            try:
                filename = _validate_dduf_entry_name(filename)
            except DDUFInvalidEntryNameError as e:
                raise DDUFExportError(f"Invalid entry name: {filename}") from e
            logger.debug(f"Adding entry '{filename}' to DDUF file")
            _dump_content_in_archive(archive, filename, content)

    # Consistency checks on the DDUF file
    if index is None:
        raise DDUFExportError("Missing required 'model_index.json' entry in DDUF file.")
    try:
        _validate_dduf_structure(index, filenames)
    except DDUFCorruptedFileError as e:
        raise DDUFExportError("Invalid DDUF file structure.") from e

    logger.info(f"Done writing DDUF file {dduf_path}")


def export_folder_as_dduf(dduf_path: str | os.PathLike, folder_path: str | os.PathLike) -> None:
    """
    Export a folder as a DDUF file.

    AUses [`export_entries_as_dduf`] under the hood.

    Args:
        dduf_path (`str` or `os.PathLike`):
            The path to the DDUF file to write.
        folder_path (`str` or `os.PathLike`):
            The path to the folder containing the diffusion model.

    Example:
        ```python
        >>> from huggingface_hub import export_folder_as_dduf
        >>> export_folder_as_dduf(dduf_path="FLUX.1-dev.dduf", folder_path="path/to/FLUX.1-dev")
        ```
    """
    folder_path = Path(folder_path)

    def _iterate_over_folder() -> Iterable[tuple[str, Path]]:
        for path in Path(folder_path).glob("**/*"):
            if not path.is_file():
                continue
            if path.suffix not in DDUF_ALLOWED_ENTRIES:
                logger.debug(f"Skipping file '{path}' (file type not allowed)")
                continue
            path_in_archive = path.relative_to(folder_path)
            if len(path_in_archive.parts) >= 3:
                logger.debug(f"Skipping file '{path}' (nested directories not allowed)")
                continue
            yield path_in_archive.as_posix(), path

    export_entries_as_dduf(dduf_path, _iterate_over_folder())


def _dump_content_in_archive(archive: zipfile.ZipFile, filename: str, content: str | os.PathLike | bytes) -> None:
    with archive.open(filename, "w", force_zip64=True) as archive_fh:
        if isinstance(content, (str, Path)):
            content_path = Path(content)
            with content_path.open("rb") as content_fh:
                shutil.copyfileobj(content_fh, archive_fh, 1024 * 1024 * 8)  # type: ignore[misc]
        elif isinstance(content, bytes):
            archive_fh.write(content)
        else:
            raise DDUFExportError(f"Invalid content type for {filename}. Must be str, Path or bytes.")


def _load_content(content: str | Path | bytes) -> bytes:
    """Load the content of an entry as bytes.

    Used only for small checks (not to dump content into archive).
    """
    if isinstance(content, (str, Path)):
        return Path(content).read_bytes()
    elif isinstance(content, bytes):
        return content
    else:
        raise DDUFExportError(f"Invalid content type. Must be str, Path or bytes. Got {type(content)}.")


def _validate_dduf_entry_name(entry_name: str) -> str:
    if "." + entry_name.split(".")[-1] not in DDUF_ALLOWED_ENTRIES:
        raise DDUFInvalidEntryNameError(f"File type not allowed: {entry_name}")
    if "\\" in entry_name:
        raise DDUFInvalidEntryNameError(f"Entry names must use UNIX separators ('/'). Got {entry_name}.")
    entry_name = entry_name.strip("/")
    if entry_name.count("/") > 1:
        raise DDUFInvalidEntryNameError(f"DDUF only supports 1 level of directory. Got {entry_name}.")
    return entry_name


def _validate_dduf_structure(index: Any, entry_names: Iterable[str]) -> None:
    """
    Consistency checks on the DDUF file structure.

    Rules:
    - The 'model_index.json' entry is required and must contain a dictionary.
    - Each folder name must correspond to an entry in 'model_index.json'.
    - Each folder must contain at least a config file ('config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'scheduler_config.json').

    Args:
        index (Any):
            The content of the 'model_index.json' entry.
        entry_names (Iterable[str]):
            The list of entry names in the DDUF file.

    Raises:
        - [`DDUFCorruptedFileError`]: If the DDUF file is corrupted (i.e. doesn't follow the DDUF format).
    """
    if not isinstance(index, dict):
        raise DDUFCorruptedFileError(f"Invalid 'model_index.json' content. Must be a dictionary. Got {type(index)}.")

    dduf_folders = {entry.split("/")[0] for entry in entry_names if "/" in entry}
    for folder in dduf_folders:
        if folder not in index:
            raise DDUFCorruptedFileError(f"Missing required entry '{folder}' in 'model_index.json'.")
        if not any(f"{folder}/{required_entry}" in entry_names for required_entry in DDUF_FOLDER_REQUIRED_ENTRIES):
            raise DDUFCorruptedFileError(
                f"Missing required file in folder '{folder}'. Must contains at least one of {DDUF_FOLDER_REQUIRED_ENTRIES}."
            )


def _get_data_offset(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    """
    Calculate the data offset for a file in a ZIP archive.

    Args:
        zf (`zipfile.ZipFile`):
            The opened ZIP file. Must be opened in read mode.
        info (`zipfile.ZipInfo`):
            The file info.

    Returns:
        int: The offset of the file data in the ZIP archive.
    """
    if zf.fp is None:
        raise DDUFCorruptedFileError("ZipFile object must be opened in read mode.")

    # Step 1: Get the local file header offset
    header_offset = info.header_offset

    # Step 2: Read the local file header
    zf.fp.seek(header_offset)
    local_file_header = zf.fp.read(30)  # Fixed-size part of the local header

    if len(local_file_header) < 30:
        raise DDUFCorruptedFileError("Incomplete local file header.")

    # Step 3: Parse the header fields to calculate the start of file data
    # Local file header: https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    filename_len = int.from_bytes(local_file_header[26:28], "little")
    extra_field_len = int.from_bytes(local_file_header[28:30], "little")

    # Data offset is after the fixed header, filename, and extra fields
    data_offset = header_offset + 30 + filename_len + extra_field_len

    return data_offset
<|endoftext|># Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Contains helpers to split tensors into shards."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .. import logging


TensorT = TypeVar("TensorT")
TensorSizeFn_T = Callable[[TensorT], int]
StorageIDFn_T = Callable[[TensorT], Any | None]

MAX_SHARD_SIZE = "5GB"
SIZE_UNITS = {
    "TB": 10**12,
    "GB": 10**9,
    "MB": 10**6,
    "KB": 10**3,
}


logger = logging.get_logger(__file__)


@dataclass
class StateDictSplit:
    is_sharded: bool = field(init=False)
    metadata: dict[str, Any]
    filename_to_tensors: dict[str, list[str]]
    tensor_to_filename: dict[str, str]

    def __post_init__(self):
        self.is_sharded = len(self.filename_to_tensors) > 1


def split_state_dict_into_shards_factory(
    state_dict: dict[str, TensorT],
    *,
    get_storage_size: TensorSizeFn_T,
    filename_pattern: str,
    get_storage_id: StorageIDFn_T = lambda tensor: None,
    max_shard_size: int | str = MAX_SHARD_SIZE,
) -> StateDictSplit:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].

    > [!WARNING]
    > If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    > size greater than `max_shard_size`.

    Args:
        state_dict (`dict[str, Tensor]`):
            The state dictionary to save.
        get_storage_size (`Callable[[Tensor], int]`):
            A function that returns the size of a tensor when saved on disk in bytes.
        get_storage_id (`Callable[[Tensor], Optional[Any]]`, *optional*):
            A function that returns a unique identifier to a tensor storage. Multiple different tensors can share the
            same underlying storage. This identifier is guaranteed to be unique and constant for this tensor's storage
            during its lifetime. Two tensor storages with non-overlapping lifetimes may have the same id.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.
    """
    storage_id_to_tensors: dict[Any, list[str]] = {}

    shard_list: list[dict[str, TensorT]] = []
    current_shard: dict[str, TensorT] = {}
    current_shard_size = 0
    total_size = 0

    if isinstance(max_shard_size, str):
        max_shard_size = parse_size_to_int(max_shard_size)

    for key, tensor in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(tensor, str):
            logger.info("Skipping tensor %s as it is a string (bnb serialization)", key)
            continue

        # If a `tensor` shares the same underlying storage as another tensor, we put `tensor` in the same `block`
        storage_id = get_storage_id(tensor)  # type: ignore[invalid-argument-type]
        if storage_id is not None:
            if storage_id in storage_id_to_tensors:
                # We skip this tensor for now and will reassign to correct shard later
                storage_id_to_tensors[storage_id].append(key)
                continue
            else:
                # This is the first tensor with this storage_id, we create a new entry
                # in the storage_id_to_tensors dict => we will assign the shard id later
                storage_id_to_tensors[storage_id] = [key]

        # Compute tensor size
        tensor_size = get_storage_size(tensor)  # type: ignore[invalid-argument-type]

        # If this tensor is bigger than the maximal size, we put it in its own shard
        if tensor_size > max_shard_size:
            total_size += tensor_size
            shard_list.append({key: tensor})
            continue

        # If this tensor is going to tip up over the maximal size, we split.
        # Current shard already has some tensors, we add it to the list of shards and create a new one.
        if current_shard_size + tensor_size > max_shard_size:
            shard_list.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        # Add the tensor to the current shard
        current_shard[key] = tensor
        current_shard_size += tensor_size
        total_size += tensor_size

    # Add the last shard
    if len(current_shard) > 0:
        shard_list.append(current_shard)
    nb_shards = len(shard_list)

    # Loop over the tensors that share the same storage and assign them together
    for storage_id, keys in storage_id_to_tensors.items():
        # Let's try to find the shard where the first tensor of this storage is and put all tensors in the same shard
        for shard in shard_list:
            if keys[0] in shard:
                for key in keys:
                    shard[key] = state_dict[key]
                break

    # If we only have one shard, we return it => no need to build the index
    if nb_shards == 1:
        filename = filename_pattern.format(suffix="")
        # Use the keys from the shard itself rather than `state_dict` directly, so that tensors
        # skipped above (e.g. string tensors from bnb serialization) are excluded from the index,
        # consistently with the multi-shard path below.
        keys = list(shard_list[0].keys())
        return StateDictSplit(
            metadata={"total_size": total_size},
            filename_to_tensors={filename: keys},
            tensor_to_filename={key: filename for key in keys},
        )

    # Now that each tensor is assigned to a shard, let's assign a filename to each shard
    tensor_name_to_filename = {}
    filename_to_tensors = {}
    for idx, shard in enumerate(shard_list):
        filename = filename_pattern.format(suffix=f"-{idx + 1:05d}-of-{nb_shards:05d}")
        for key in shard:
            tensor_name_to_filename[key] = filename
        filename_to_tensors[filename] = list(shard.keys())

    # Build the index and return
    return StateDictSplit(
        metadata={"total_size": total_size},
        filename_to_tensors=filename_to_tensors,
        tensor_to_filename=tensor_name_to_filename,
    )


def parse_size_to_int(size_as_str: str) -> int:
    """
    Parse a size expressed as a string with digits and unit (like `"5MB"`) to an integer (in bytes).

    Supported units are "TB", "GB", "MB", "KB".

    Args:
        size_as_str (`str`): The size to convert.

    Example:

    ```py
    >>> parse_size_to_int("5MB")
    5000000
    ```
    """
    size_as_str = size_as_str.strip()

    # Parse unit
    unit = size_as_str[-2:].upper()
    if unit not in SIZE_UNITS:
        raise ValueError(f"Unit '{unit}' not supported. Supported units are TB, GB, MB, KB. Got '{size_as_str}'.")
    multiplier = SIZE_UNITS[unit]

    # Parse value
    try:
        value = float(size_as_str[:-2].strip())
    except ValueError as e:
        raise ValueError(f"Could not parse the size value from '{size_as_str}': {e}") from e

    return int(value * multiplier)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PCX file handling
#
# This format was originally used by ZSoft's popular PaintBrush
# program for the IBM PC.  It is also supported by many MS-DOS and
# Windows applications, including the Windows PaintBrush program in
# Windows 3.
#
# history:
# 1995-09-01 fl   Created
# 1996-05-20 fl   Fixed RGB support
# 1997-01-03 fl   Fixed 2-bit and 4-bit support
# 1999-02-03 fl   Fixed 8-bit support (broken in 1.0b1)
# 1999-02-07 fl   Added write support
# 2002-06-09 fl   Made 2-bit and 4-bit support a bit more robust
# 2002-07-30 fl   Seek from to current position, not beginning of file
# 2003-06-03 fl   Extract DPI settings (info["dpi"])
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
import logging
from typing import IO

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16

logger = logging.getLogger(__name__)


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 2 and prefix[0] == 10 and prefix[1] in [0, 2, 3, 5]


##
# Image plugin for Paintbrush images.


class PcxImageFile(ImageFile.ImageFile):
    format = "PCX"
    format_description = "Paintbrush"

    def _open(self) -> None:
        # header
        assert self.fp is not None

        s = self.fp.read(68)
        if not _accept(s):
            msg = "not a PCX file"
            raise SyntaxError(msg)

        # image
        bbox = i16(s, 4), i16(s, 6), i16(s, 8) + 1, i16(s, 10) + 1
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            msg = "bad PCX image size"
            raise SyntaxError(msg)
        logger.debug("BBox: %s %s %s %s", *bbox)

        offset = self.fp.tell() + 60

        # format
        version = s[1]
        bits = s[3]
        planes = s[65]
        provided_stride = i16(s, 66)
        logger.debug(
            "PCX version %s, bits %s, planes %s, stride %s",
            version,
            bits,
            planes,
            provided_stride,
        )

        self.info["dpi"] = i16(s, 12), i16(s, 14)

        if bits == 1 and planes == 1:
            mode = rawmode = "1"

        elif bits == 1 and planes in (2, 4):
            mode = "P"
            rawmode = f"P;{planes}L"
            self.palette = ImagePalette.raw("RGB", s[16:64])

        elif version == 5 and bits == 8 and planes == 1:
            mode = rawmode = "L"
            # FIXME: hey, this doesn't work with the incremental loader !!!
            self.fp.seek(-769, io.SEEK_END)
            s = self.fp.read(769)
            if len(s) == 769 and s[0] == 12:
                # check if the palette is linear grayscale
                for i in range(256):
                    if s[i * 3 + 1 : i * 3 + 4] != o8(i) * 3:
                        mode = rawmode = "P"
                        break
                if mode == "P":
                    self.palette = ImagePalette.raw("RGB", s[1:])

        elif version == 5 and bits == 8 and planes == 3:
            mode = "RGB"
            rawmode = "RGB;L"

        else:
            msg = "unknown PCX mode"
            raise OSError(msg)

        self._mode = mode
        self._size = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Don't trust the passed in stride.
        # Calculate the approximate position for ourselves.
        # CVE-2020-35653
        stride = (self._size[0] * bits + 7) // 8

        # While the specification states that this must be even,
        # not all images follow this
        if provided_stride != stride:
            stride += stride % 2

        bbox = (0, 0) + self.size
        logger.debug("size: %sx%s", *self.size)

        self.tile = [ImageFile._Tile("pcx", bbox, offset, (rawmode, planes * stride))]


# --------------------------------------------------------------------
# save PCX files


SAVE = {
    # mode: (version, bits, planes, raw mode)
    "1": (2, 1, 1, "1"),
    "L": (5, 8, 1, "L"),
    "P": (5, 8, 1, "P"),
    "RGB": (5, 8, 3, "RGB;L"),
}


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.width == 0 or im.height == 0:
        msg = "Cannot write empty image as PCX"
        raise ValueError(msg)

    try:
        version, bits, planes, rawmode = SAVE[im.mode]
    except KeyError as e:
        msg = f"Cannot save {im.mode} images as PCX"
        raise ValueError(msg) from e

    # bytes per plane
    stride = (im.size[0] * bits + 7) // 8
    # stride should be even
    stride += stride % 2
    # Stride needs to be kept in sync with the PcxEncode.c version.
    # Ideally it should be passed in in the state, but the bytes value
    # gets overwritten.

    logger.debug(
        "PcxImagePlugin._save: xwidth: %d, bits: %d, stride: %d",
        im.size[0],
        bits,
        stride,
    )

    # under windows, we could determine the current screen size with
    # "Image.core.display_mode()[1]", but I think that's overkill...

    screen = im.size

    dpi = 100, 100

    # PCX header
    fp.write(
        o8(10)
        + o8(version)
        + o8(1)
        + o8(bits)
        + o16(0)
        + o16(0)
        + o16(im.size[0] - 1)
        + o16(im.size[1] - 1)
        + o16(dpi[0])
        + o16(dpi[1])
        + b"\0" * 24
        + b"\xff" * 24
        + b"\0"
        + o8(planes)
        + o16(stride)
        + o16(1)
        + o16(screen[0])
        + o16(screen[1])
        + b"\0" * 54
    )

    assert fp.tell() == 128

    ImageFile._save(
        im, fp, [ImageFile._Tile("pcx", (0, 0) + im.size, 0, (rawmode, bits * planes))]
    )

    if im.mode == "P":
        # colour palette
        fp.write(o8(12))
        palette = im.im.getpalette("RGB", "RGB")
        palette += b"\x00" * (768 - len(palette))
        fp.write(palette)  # 768 bytes
    elif im.mode == "L":
        # grayscale palette
        fp.write(o8(12))
        for i in range(256):
            fp.write(o8(i) * 3)


# --------------------------------------------------------------------
# registry


Image.register_open(PcxImageFile.format, PcxImageFile, _accept)
Image.register_save(PcxImageFile.format, _save)

Image.register_extension(PcxImageFile.format, ".pcx")

Image.register_mime(PcxImageFile.format, "image/x-pcx")
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# bitmap distribution font (bdf) file parser
#
# history:
# 1996-05-16 fl   created (as bdf2pil)
# 1997-08-25 fl   converted to FontFile driver
# 2001-05-25 fl   removed bogus __init__ call
# 2002-11-20 fl   robustification (from Kevin Cazabon, Dmitry Vasiliev)
# 2003-04-22 fl   more robustification (from Graham Dumpleton)
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1997-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

"""
Parse X Bitmap Distribution Format (BDF)
"""

from __future__ import annotations

from typing import BinaryIO

from . import FontFile, Image


def bdf_char(
    f: BinaryIO,
) -> (
    tuple[
        str,
        int,
        tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]],
        Image.Image,
    ]
    | None
):
    # skip to STARTCHAR
    while True:
        s = f.readline()
        if not s:
            return None
        if s.startswith(b"STARTCHAR"):
            break
    id = s[9:].strip().decode("ascii")

    # load symbol properties
    props = {}
    while True:
        s = f.readline()
        if not s or s.startswith(b"BITMAP"):
            break
        i = s.find(b" ")
        props[s[:i].decode("ascii")] = s[i + 1 : -1].decode("ascii")

    # load bitmap
    bitmap = bytearray()
    while True:
        s = f.readline()
        if not s or s.startswith(b"ENDCHAR"):
            break
        bitmap += s[:-1]

    # The word BBX
    # followed by the width in x (BBw), height in y (BBh),
    # and x and y displacement (BBxoff0, BByoff0)
    # of the lower left corner from the origin of the character.
    width, height, x_disp, y_disp = (int(p) for p in props["BBX"].split())
    Image._decompression_bomb_check((width, height))

    # The word DWIDTH
    # followed by the width in x and y of the character in device pixels.
    dwx, dwy = (int(p) for p in props["DWIDTH"].split())

    bbox = (
        (dwx, dwy),
        (x_disp, -y_disp - height, width + x_disp, -y_disp),
        (0, 0, width, height),
    )

    try:
        im = Image.frombytes("1", (width, height), bitmap, "hex", "1")
    except ValueError:
        # deal with zero-width characters
        im = Image.new("1", (width, height))

    return id, int(props["ENCODING"]), bbox, im


class BdfFontFile(FontFile.FontFile):
    """Font file plugin for the X11 BDF format."""

    def __init__(self, fp: BinaryIO) -> None:
        super().__init__()

        s = fp.readline()
        if not s.startswith(b"STARTFONT 2.1"):
            msg = "not a valid BDF file"
            raise SyntaxError(msg)

        props = {}
        comments = []

        while True:
            s = fp.readline()
            if not s or s.startswith(b"ENDPROPERTIES"):
                break
            i = s.find(b" ")
            props[s[:i].decode("ascii")] = s[i + 1 : -1].decode("ascii")
            if s[:i] in [b"COMMENT", b"COPYRIGHT"]:
                if s.find(b"LogicalFontDescription") < 0:
                    comments.append(s[i + 1 : -1].decode("ascii"))

        while True:
            c = bdf_char(fp)
            if not c:
                break
            id, ch, (xy, dst, src), im = c
            if 0 <= ch < len(self.glyph):
                self.glyph[ch] = xy, dst, src, im
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# XV Thumbnail file handler by Charles E. "Gene" Cash
# (gcash@magicnet.net)
#
# see xvcolor.c and xvbrowse.c in the sources to John Bradley's XV,
# available from ftp://ftp.cis.upenn.edu/pub/xv/
#
# history:
# 98-08-15 cec  created (b/w only)
# 98-12-09 cec  added color palette
# 98-12-28 fl   added to PIL (with only a few very minor modifications)
#
# To do:
# FIXME: make save work (this requires quantization support)
#
from __future__ import annotations

from . import Image, ImageFile, ImagePalette
from ._binary import o8

_MAGIC = b"P7 332"

# standard color palette for thumbnails (RGB332)
PALETTE = b""
for r in range(8):
    for g in range(8):
        for b in range(4):
            PALETTE = PALETTE + (
                o8((r * 255) // 7) + o8((g * 255) // 7) + o8((b * 255) // 3)
            )


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(_MAGIC)


##
# Image plugin for XV thumbnail images.


class XVThumbImageFile(ImageFile.ImageFile):
    format = "XVThumb"
    format_description = "XV thumbnail image"

    def _open(self) -> None:
        # check magic
        assert self.fp is not None

        if not _accept(self.fp.read(6)):
            msg = "not an XV thumbnail file"
            raise SyntaxError(msg)

        # Skip to beginning of next line
        self.fp.readline()

        # skip info comments
        while True:
            s = self.fp.readline()
            if not s:
                msg = "Unexpected EOF reading XV thumbnail file"
                raise SyntaxError(msg)
            if s[0] != 35:  # ie. when not a comment: '#'
                break

        # parse header line (already read)
        w, h = s.strip().split(maxsplit=2)[:2]

        self._mode = "P"
        self._size = int(w), int(h)

        self.palette = ImagePalette.raw("RGB", PALETTE)

        self.tile = [
            ImageFile._Tile("raw", (0, 0) + self.size, self.fp.tell(), self.mode)
        ]


# --------------------------------------------------------------------

Image.register_open(XVThumbImageFile.format, XVThumbImageFile, _accept)
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# drawing interface operations
#
# History:
# 1996-04-13 fl   Created (experimental)
# 1996-08-07 fl   Filled polygons, ellipses.
# 1996-08-13 fl   Added text support
# 1998-06-28 fl   Handle I and F images
# 1998-12-29 fl   Added arc; use arc primitive to draw ellipses
# 1999-01-10 fl   Added shape stuff (experimental)
# 1999-02-06 fl   Added bitmap support
# 1999-02-11 fl   Changed all primitives to take options
# 1999-02-20 fl   Fixed backwards compatibility
# 2000-10-12 fl   Copy on write, when necessary
# 2001-02-18 fl   Use default ink for bitmap/text also in fill mode
# 2002-10-24 fl   Added support for CSS-style color strings
# 2002-12-10 fl   Added experimental support for RGBA-on-RGB drawing
# 2002-12-11 fl   Refactored low-level drawing API (work in progress)
# 2004-08-26 fl   Made Draw() a factory function, added getdraw() support
# 2004-09-04 fl   Added width support to line primitive
# 2004-09-10 fl   Added font mode handling
# 2006-06-19 fl   Added font bearing support (getmask2)
#
# Copyright (c) 1997-2006 by Secret Labs AB
# Copyright (c) 1996-2006 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import cast

from . import Image, ImageColor, ImageFont, ImageText

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, AnyStr

    from . import ImageDraw2
    from ._typing import Coords, _Ink

# experimental access to the outline API
Outline: Callable[[], Image.core._Outline] = Image.core.outline

"""
A simple 2D drawing interface for PIL images.
<p>
Application code should use the <b>Draw</b> factory, instead of
directly.
"""


class ImageDraw:
    font: ImageFont.BaseImageFont | None = None

    def __init__(self, im: Image.Image, mode: str | None = None) -> None:
        """
        Create a drawing instance.

        :param im: The image to draw in.
        :param mode: Optional mode to use for color values.  For RGB
           images, this argument can be RGB or RGBA (to blend the
           drawing into the image).  For all other modes, this argument
           must be the same as the image mode.  If omitted, the mode
           defaults to the mode of the image.
        """
        im._ensure_mutable()
        blend = 0
        if mode is None:
            mode = im.mode
        if mode != im.mode:
            if mode == "RGBA" and im.mode == "RGB":
                blend = 1
            else:
                msg = "mode mismatch"
                raise ValueError(msg)
        if mode == "P":
            self.palette = im.palette
        else:
            self.palette = None
        self._image = im
        self.im = im.im
        self.draw = Image.core.draw(self.im, blend)
        self.mode = mode
        if mode in ("I", "F"):
            self.ink = self.draw.draw_ink(1)
        else:
            self.ink = self.draw.draw_ink(-1)
        if mode in ("1", "P", "I", "F"):
            # FIXME: fix Fill2 to properly support matte for I+F images
            self.fontmode = "1"
        else:
            self.fontmode = "L"  # aliasing is okay for other modes
        self.fill = False

    def getfont(
        self,
    ) -> ImageFont.BaseImageFont:
        """
        Get the current default font.

        To set the default font for this ImageDraw instance::

            from PIL import ImageDraw, ImageFont
            draw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        To set the default font for all future ImageDraw instances::

            from PIL import ImageDraw, ImageFont
            ImageDraw.ImageDraw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")

        If the current default font is ``None``,
        it is initialized with ``ImageFont.load_default()``.

        :returns: An image font."""
        if not self.font:
            # FIXME: should add a font repository
            self.font = ImageFont.load_default()
        return self.font

    def _getfont(self, font_size: float | None) -> ImageFont.BaseImageFont:
        if font_size is not None:
            return ImageFont.load_default(font_size)
        else:
            return self.getfont()

    def _getink(
        self, ink: _Ink | None, fill: _Ink | None = None
    ) -> tuple[int | None, int | None]:
        result_ink = None
        result_fill = None
        if ink is None and fill is None:
            if self.fill:
                result_fill = self.ink
            else:
                result_ink = self.ink
        else:
            if ink is not None:
                if isinstance(ink, str):
                    ink = ImageColor.getcolor(ink, self.mode)
                if self.palette and isinstance(ink, tuple):
                    ink = self.palette.getcolor(ink, self._image)
                result_ink = self.draw.draw_ink(ink)
            if fill is not None:
                if isinstance(fill, str):
                    fill = ImageColor.getcolor(fill, self.mode)
                if self.palette and isinstance(fill, tuple):
                    fill = self.palette.getcolor(fill, self._image)
                result_fill = self.draw.draw_ink(fill)
        return result_ink, result_fill

    def arc(
        self,
        xy: Coords,
        start: float,
        end: float,
        fill: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw an arc."""
        ink, fill = self._getink(fill)
        if ink is not None and width != 0:
            self.draw.draw_arc(xy, start, end, ink, width)

    def bitmap(
        self, xy: Sequence[int], bitmap: Image.Image, fill: _Ink | None = None
    ) -> None:
        """Draw a bitmap."""
        bitmap.load()
        ink, fill = self._getink(fill)
        if ink is None:
            ink = fill
        if ink is not None:
            self.draw.draw_bitmap(xy, bitmap.im, ink)

    def chord(
        self,
        xy: Coords,
        start: float,
        end: float,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a chord."""
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_chord(xy, start, end, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            self.draw.draw_chord(xy, start, end, ink, 0, width)

    def ellipse(
        self,
        xy: Coords,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw an ellipse."""
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_ellipse(xy, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            self.draw.draw_ellipse(xy, ink, 0, width)

    def circle(
        self,
        xy: Sequence[float],
        radius: float,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a circle given center coordinates and a radius."""
        ellipse_xy = (xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius)
        self.ellipse(ellipse_xy, fill, outline, width)

    def line(
        self,
        xy: Coords,
        fill: _Ink | None = None,
        width: int = 1,
        joint: str | None = None,
    ) -> None:
        """Draw a line, or a connected sequence of line segments."""
        ink = self._getink(fill)[0]
        if ink is not None and width != 0:
            self.draw.draw_lines(xy, ink, width)
            if joint == "curve" and width > 4:
                points: Sequence[Sequence[float]]
                if isinstance(xy[0], (list, tuple)):
                    points = cast(Sequence[Sequence[float]], xy)
                else:
                    points = [
                        cast(Sequence[float], tuple(xy[i : i + 2]))
                        for i in range(0, len(xy), 2)
                    ]
                for i in range(1, len(points) - 1):
                    point = points[i]
                    angles = [
                        math.degrees(math.atan2(end[0] - start[0], start[1] - end[1]))
                        % 360
                        for start, end in (
                            (points[i - 1], point),
                            (point, points[i + 1]),
                        )
                    ]
                    if angles[0] == angles[1]:
                        # This is a straight line, so no joint is required
                        continue

                    def coord_at_angle(
                        coord: Sequence[float], angle: float
                    ) -> tuple[float, ...]:
                        x, y = coord
                        angle -= 90
                        distance = width / 2 - 1
                        return tuple(
                            p + (math.floor(p_d) if p_d > 0 else math.ceil(p_d))
                            for p, p_d in (
                                (x, distance * math.cos(math.radians(angle))),
                                (y, distance * math.sin(math.radians(angle))),
                            )
                        )

                    flipped = (
                        angles[1] > angles[0] and angles[1] - 180 > angles[0]
                    ) or (angles[1] < angles[0] and angles[1] + 180 > angles[0])
                    coords = [
                        (point[0] - width / 2 + 1, point[1] - width / 2 + 1),
                        (point[0] + width / 2 - 1, point[1] + width / 2 - 1),
                    ]
                    if flipped:
                        start, end = (angles[1] + 90, angles[0] + 90)
                    else:
                        start, end = (angles[0] - 90, angles[1] - 90)
                    self.pieslice(coords, start - 90, end - 90, fill)

                    if width > 8:
                        # Cover potential gaps between the line and the joint
                        if flipped:
                            gap_coords = [
                                coord_at_angle(point, angles[0] + 90),
                                point,
                                coord_at_angle(point, angles[1] + 90),
                            ]
                        else:
                            gap_coords = [
                                coord_at_angle(point, angles[0] - 90),
                                point,
                                coord_at_angle(point, angles[1] - 90),
                            ]
                        self.line(gap_coords, fill, width=3)

    def shape(
        self,
        shape: Image.core._Outline,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
    ) -> None:
        """(Experimental) Draw a shape."""
        shape.close()
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_outline(shape, fill_ink, 1)
        if ink is not None and ink != fill_ink:
            self.draw.draw_outline(shape, ink, 0)

    def pieslice(
        self,
        xy: Coords,
        start: float,
        end: float,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a pieslice."""
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_pieslice(xy, start, end, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            self.draw.draw_pieslice(xy, start, end, ink, 0, width)

    def point(self, xy: Coords, fill: _Ink | None = None) -> None:
        """Draw one or more individual pixels."""
        ink, fill = self._getink(fill)
        if ink is not None:
            self.draw.draw_points(xy, ink)

    def polygon(
        self,
        xy: Coords,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a polygon."""
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_polygon(xy, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            if width == 1:
                self.draw.draw_polygon(xy, ink, 0, width)
            elif self.im is not None:
                # To avoid expanding the polygon outwards,
                # use the fill as a mask
                mask = Image.new("1", self.im.size)
                mask_ink = self._getink(1)[0]
                draw = Draw(mask)
                draw.draw.draw_polygon(xy, mask_ink, 1)

                self.draw.draw_polygon(xy, ink, 0, width * 2 - 1, mask.im)

    def regular_polygon(
        self,
        bounding_circle: Sequence[Sequence[float] | float],
        n_sides: int,
        rotation: float = 0,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a regular polygon."""
        xy = _compute_regular_polygon_vertices(bounding_circle, n_sides, rotation)
        self.polygon(xy, fill, outline, width)

    def rectangle(
        self,
        xy: Coords,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
    ) -> None:
        """Draw a rectangle."""
        ink, fill_ink = self._getink(outline, fill)
        if fill_ink is not None:
            self.draw.draw_rectangle(xy, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            self.draw.draw_rectangle(xy, ink, 0, width)

    def rounded_rectangle(
        self,
        xy: Coords,
        radius: float = 0,
        fill: _Ink | None = None,
        outline: _Ink | None = None,
        width: int = 1,
        *,
        corners: tuple[bool, bool, bool, bool] | None = None,
    ) -> None:
        """Draw a rounded rectangle."""
        if isinstance(xy[0], (list, tuple)):
            (x0, y0), (x1, y1) = cast(Sequence[Sequence[float]], xy)
        else:
            x0, y0, x1, y1 = cast(Sequence[float], xy)
        if x1 < x0:
            msg = "x1 must be greater than or equal to x0"
            raise ValueError(msg)
        if y1 < y0:
            msg = "y1 must be greater than or equal to y0"
            raise ValueError(msg)
        if corners is None:
            corners = (True, True, True, True)

        d = min(x1 - x0, y1 - y0, radius * 2)

        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        full_x, full_y = False, False
        if all(corners):
            full_x = d >= x1 - x0 - 1
            if full_x:
                # The two left and two right corners are joined
                d = x1 - x0
            full_y = d >= y1 - y0 - 1
            if full_y:
                # The two top and two bottom corners are joined
                d = y1 - y0
            if full_x and full_y:
                # If all corners are joined, that is a circle
                return self.ellipse(xy, fill, outline, width)

        if d == 0 or not any(corners):
            # If the corners have no curve,
            # or there are no corners,
            # that is a rectangle
            return self.rectangle(xy, fill, outline, width)

        r = int(d // 2)
        ink, fill_ink = self._getink(outline, fill)

        def draw_corners(pieslice: bool) -> None:
            parts: tuple[tuple[tuple[float, float, float, float], int, int], ...]
            if full_x:
                # Draw top and bottom halves
                parts = (
                    ((x0, y0, x0 + d, y0 + d), 180, 360),
                    ((x0, y1 - d, x0 + d, y1), 0, 180),
                )
            elif full_y:
                # Draw left and right halves
                parts = (
                    ((x0, y0, x0 + d, y0 + d), 90, 270),
                    ((x1 - d, y0, x1, y0 + d), 270, 90),
                )
            else:
                # Draw four separate corners
                parts = tuple(
                    part
                    for i, part in enumerate(
                        (
                            ((x0, y0, x0 + d, y0 + d), 180, 270),
                            ((x1 - d, y0, x1, y0 + d), 270, 360),
                            ((x1 - d, y1 - d, x1, y1), 0, 90),
                            ((x0, y1 - d, x0 + d, y1), 90, 180),
                        )
                    )
                    if corners[i]
                )
            for part in parts:
                if pieslice:
                    self.draw.draw_pieslice(*(part + (fill_ink, 1)))
                else:
                    self.draw.draw_arc(*(part + (ink, width)))

        if fill_ink is not None:
            draw_corners(True)

            if full_x:
                self.draw.draw_rectangle((x0, y0 + r + 1, x1, y1 - r - 1), fill_ink, 1)
            elif x1 - r - 1 >= x0 + r + 1:
                self.draw.draw_rectangle((x0 + r + 1, y0, x1 - r - 1, y1), fill_ink, 1)
            if not full_x and not full_y:
                left = [x0, y0, x0 + r, y1]
                if corners[0]:
                    left[1] += r + 1
                if corners[3]:
                    left[3] -= r + 1
                self.draw.draw_rectangle(left, fill_ink, 1)

                right = [x1 - r, y0, x1, y1]
                if corners[1]:
                    right[1] += r + 1
                if corners[2]:
                    right[3] -= r + 1
                self.draw.draw_rectangle(right, fill_ink, 1)
        if ink is not None and ink != fill_ink and width != 0:
            draw_corners(False)

            if not full_x:
                top = [x0, y0, x1, y0 + width - 1]
                if corners[0]:
                    top[0] += r + 1
                if corners[1]:
                    top[2] -= r + 1
                self.draw.draw_rectangle(top, ink, 1)

                bottom = [x0, y1 - width + 1, x1, y1]
                if corners[3]:
                    bottom[0] += r + 1
                if corners[2]:
                    bottom[2] -= r + 1
                self.draw.draw_rectangle(bottom, ink, 1)
            if not full_y:
                left = [x0, y0, x0 + width - 1, y1]
                if corners[0]:
                    left[1] += r + 1
                if corners[3]:
                    left[3] -= r + 1
                self.draw.draw_rectangle(left, ink, 1)

                right = [x1 - width + 1, y0, x1, y1]
                if corners[1]:
                    right[1] += r + 1
                if corners[2]:
                    right[3] -= r + 1
                self.draw.draw_rectangle(right, ink, 1)

    def text(
        self,
        xy: tuple[float, float],
        text: AnyStr | ImageText.Text[AnyStr],
        fill: _Ink | None = None,
        font: ImageFont.BaseImageFont | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        stroke_fill: _Ink | None = None,
        embedded_color: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Draw text."""
        if isinstance(text, ImageText.Text):
            image_text = text
        else:
            if font is None:
                font = self._getfont(kwargs.get("font_size"))
            image_text = ImageText.Text(
                text, font, self.mode, spacing, direction, features, language
            )
            if embedded_color:
                image_text.embed_color()
            if stroke_width:
                image_text.stroke(stroke_width, stroke_fill)

        def getink(fill: _Ink | None) -> int:
            ink, fill_ink = self._getink(fill)
            if ink is None:
                assert fill_ink is not None
                return fill_ink
            return ink

        ink = getink(fill)
        if ink is None:
            return

        stroke_ink = None
        if image_text.stroke_width:
            stroke_ink = (
                getink(image_text.stroke_fill)
                if image_text.stroke_fill is not None
                else ink
            )

        for line in image_text._split(xy, anchor, align):

            def draw_text(ink: int, stroke_width: float = 0) -> None:
                mode = self.fontmode
                if stroke_width == 0 and embedded_color:
                    mode = "RGBA"
                x = int(line.x)
                y = int(line.y)
                start = (math.modf(line.x)[0], math.modf(line.y)[0])
                if isinstance(image_text.font, ImageFont.FreeTypeFont):
                    mask, offset = image_text.font.getmask2(
                        line.text,
                        mode,
                        direction,
                        features,
                        language,
                        stroke_width,
                        line.anchor,
                        ink,
                        start,
                        stroke_filled=True,
                        *args,
                        **kwargs,
                    )
                    x += offset[0]
                    y += offset[1]
                else:
                    try:
                        mask = image_text.font.getmask(
                            line.text,
                            mode,
                            direction,
                            features,
                            language,
                            stroke_width,
                            line.anchor,
                            ink,
                            start=start,
                            *args,
                            **kwargs,
                        )
                    except TypeError:
                        mask = image_text.font.getmask(line.text)
                if mode == "RGBA":
                    # image_text.font.getmask2(mode="RGBA")
                    # returns color in RGB bands and mask in A
                    # extract mask and set text alpha
                    color, mask = mask, mask.getband(3)
                    ink_alpha = struct.pack("i", ink)[3]
                    color.fillband(3, ink_alpha)
                    if self.im is not None:
                        self.im.paste(
                            color, (x, y, x + mask.size[0], y + mask.size[1]), mask
                        )
                else:
                    self.draw.draw_bitmap((x, y), mask, ink)

            if stroke_ink is not None:
                # Draw stroked text
                draw_text(stroke_ink, image_text.stroke_width)

                # Draw normal text
                if ink != stroke_ink:
                    draw_text(ink)
            else:
                # Only draw normal text
                draw_text(ink)

    def multiline_text(
        self,
        xy: tuple[float, float],
        text: AnyStr,
        fill: _Ink | None = None,
        font: ImageFont.BaseImageFont | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        stroke_fill: _Ink | None = None,
        embedded_color: bool = False,
        *,
        font_size: float | None = None,
    ) -> None:
        return self.text(
            xy,
            text,
            fill,
            font,
            anchor,
            spacing,
            align,
            direction,
            features,
            language,
            stroke_width,
            stroke_fill,
            embedded_color,
            font_size=font_size,
        )

    def textlength(
        self,
        text: AnyStr,
        font: ImageFont.BaseImageFont | None = None,
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        embedded_color: bool = False,
        *,
        font_size: float | None = None,
    ) -> float:
        """Get the length of a given string, in pixels with 1/64 precision."""
        if font is None:
            font = self._getfont(font_size)
        image_text = ImageText.Text(
            text,
            font,
            self.mode,
            direction=direction,
            features=features,
            language=language,
        )
        if embedded_color:
            image_text.embed_color()
        return image_text.get_length()

    def textbbox(
        self,
        xy: tuple[float, float],
        text: AnyStr,
        font: ImageFont.BaseImageFont | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        embedded_color: bool = False,
        *,
        font_size: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Get the bounding box of a given string, in pixels."""
        if font is None:
            font = self._getfont(font_size)
        image_text = ImageText.Text(
            text, font, self.mode, spacing, direction, features, language
        )
        if embedded_color:
            image_text.embed_color()
        if stroke_width:
            image_text.stroke(stroke_width)
        return image_text.get_bbox(xy, anchor, align)

    def multiline_textbbox(
        self,
        xy: tuple[float, float],
        text: AnyStr,
        font: ImageFont.BaseImageFont | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        embedded_color: bool = False,
        *,
        font_size: float | None = None,
    ) -> tuple[float, float, float, float]:
        return self.textbbox(
            xy,
            text,
            font,
            anchor,
            spacing,
            align,
            direction,
            features,
            language,
            stroke_width,
            embedded_color,
            font_size=font_size,
        )


def Draw(im: Image.Image, mode: str | None = None) -> ImageDraw:
    """
    A simple 2D drawing interface for PIL images.

    :param im: The image to draw in.
    :param mode: Optional mode to use for color values.  For RGB
       images, this argument can be RGB or RGBA (to blend the
       drawing into the image).  For all other modes, this argument
       must be the same as the image mode.  If omitted, the mode
       defaults to the mode of the image.
    """
    try:
        return getattr(im, "getdraw")(mode)
    except AttributeError:
        return ImageDraw(im, mode)


def getdraw(im: Image.Image | None = None) -> tuple[ImageDraw2.Draw | None, ModuleType]:
    """
    :param im: The image to draw in.
    :returns: A (drawing context, drawing resource factory) tuple.
    """
    from . import ImageDraw2

    draw = ImageDraw2.Draw(im) if im is not None else None
    return draw, ImageDraw2


def floodfill(
    image: Image.Image,
    xy: tuple[int, int],
    value: float | tuple[int, ...],
    border: float | tuple[int, ...] | None = None,
    thresh: float = 0,
) -> None:
    """
    .. warning:: This method is experimental.

    Fills a bounded region with a given color.

    :param image: Target image.
    :param xy: Seed position (a 2-item coordinate tuple). See
        :ref:`coordinate-system`.
    :param value: Fill color.
    :param border: Optional border value.  If given, the region consists of
        pixels with a color different from the border color.  If not given,
        the region consists of pixels having the same color as the seed
        pixel.
    :param thresh: Optional threshold value which specifies a maximum
        tolerable difference of a pixel value from the 'background' in
        order for it to be replaced. Useful for filling regions of
        non-homogeneous, but similar, colors.
    """
    # based on an implementation by Eric S. Raymond
    # amended by yo1995 @20180806
    pixel = image.load()
    assert pixel is not None
    x, y = xy
    try:
        background = pixel[x, y]
        if _color_diff(value, background) <= thresh:
            return  # seed point already has fill color
        pixel[x, y] = value
    except (ValueError, IndexError):
        return  # seed point outside image
    edge = {(x, y)}
    # use a set to keep record of current and previous edge pixels
    # to reduce memory consumption
    full_edge = set()
    while edge:
        new_edge = set()
        for x, y in edge:  # 4 adjacent method
            for s, t in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # If already processed, or if a coordinate is negative, skip
                if (s, t) in full_edge or s < 0 or t < 0:
                    continue
                try:
                    p = pixel[s, t]
                except (ValueError, IndexError):
                    pass
                else:
                    full_edge.add((s, t))
                    if border is None:
                        fill = _color_diff(p, background) <= thresh
                    else:
                        fill = p not in (value, border)
                    if fill:
                        pixel[s, t] = value
                        new_edge.add((s, t))
        full_edge = edge  # discard pixels processed
        edge = new_edge


def _compute_regular_polygon_vertices(
    bounding_circle: Sequence[Sequence[float] | float], n_sides: int, rotation: float
) -> list[tuple[float, float]]:
    """
    Generate a list of vertices for a 2D regular polygon.

    :param bounding_circle: The bounding circle is a sequence defined
        by a point and radius. The polygon is inscribed in this circle.
        (e.g. ``bounding_circle=(x, y, r)`` or ``((x, y), r)``)
    :param n_sides: Number of sides
        (e.g. ``n_sides=3`` for a triangle, ``6`` for a hexagon)
    :param rotation: Apply an arbitrary rotation to the polygon
        (e.g. ``rotation=90``, applies a 90 degree rotation)
    :return: List of regular polygon vertices
        (e.g. ``[(25, 50), (50, 50), (50, 25), (25, 25)]``)

    How are the vertices computed?
    1. Compute the following variables
        - theta: Angle between the apothem & the nearest polygon vertex
        - side_length: Length of each polygon edge
        - centroid: Center of bounding circle (1st, 2nd elements of bounding_circle)
        - polygon_radius: Polygon radius (last element of bounding_circle)
        - angles: Location of each polygon vertex in polar grid
            (e.g. A square with 0 degree rotation => [225.0, 315.0, 45.0, 135.0])

    2. For each angle in angles, get the polygon vertex at that angle
        The vertex is computed using the equation below.
            X= xcos(φ) + ysin(φ)
            Y= −xsin(φ) + ycos(φ)

        Note:
            φ = angle in degrees
            x = 0
            y = polygon_radius

        The formula above assumes rotation around the origin.
        In our case, we are rotating around the centroid.
        To account for this, we use the formula below
            X = xcos(φ) + ysin(φ) + centroid_x
            Y = −xsin(φ) + ycos(φ) + centroid_y
    """
    # 1. Error Handling
    # 1.1 Check `n_sides` has an appropriate value
    if not isinstance(n_sides, int):
        msg = "n_sides should be an int"  # type: ignore[unreachable]
        raise TypeError(msg)
    if n_sides < 3:
        msg = "n_sides should be an int > 2"
        raise ValueError(msg)

    # 1.2 Check `bounding_circle` has an appropriate value
    if not isinstance(bounding_circle, (list, tuple)):
        msg = "bounding_circle should be a sequence"
        raise TypeError(msg)

    if len(bounding_circle) == 3:
        if not all(isinstance(i, (int, float)) for i in bounding_circle):
            msg = "bounding_circle should only contain numeric data"
            raise ValueError(msg)

        *centroid, polygon_radius = cast(list[float], list(bounding_circle))
    elif len(bounding_circle) == 2 and isinstance(bounding_circle[0], (list, tuple)):
        if not all(
            isinstance(i, (int, float)) for i in bounding_circle[0]
        ) or not isinstance(bounding_circle[1], (int, float)):
            msg = "bounding_circle should only contain numeric data"
            raise ValueError(msg)

        if len(bounding_circle[0]) != 2:
            msg = "bounding_circle centre should contain 2D coordinates (e.g. (x, y))"
            raise ValueError(msg)

        centroid = cast(list[float], list(bounding_circle[0]))
        polygon_radius = cast(float, bounding_circle[1])
    else:
        msg = (
            "bounding_circle should contain 2D coordinates "
            "and a radius (e.g. (x, y, r) or ((x, y), r) )"
        )
        raise ValueError(msg)

    if polygon_radius <= 0:
        msg = "bounding_circle radius should be > 0"
        raise ValueError(msg)

    # 1.3 Check `rotation` has an appropriate value
    if not isinstance(rotation, (int, float)):
        msg = "rotation should be an int or float"  # type: ignore[unreachable]
        raise ValueError(msg)

    # 2. Define Helper Functions
    def _apply_rotation(point: list[float], degrees: float) -> tuple[float, float]:
        return (
            round(
                point[0] * math.cos(math.radians(360 - degrees))
                - point[1] * math.sin(math.radians(360 - degrees))
                + centroid[0],
                2,
            ),
            round(
                point[1] * math.cos(math.radians(360 - degrees))
                + point[0] * math.sin(math.radians(360 - degrees))
                + centroid[1],
                2,
            ),
        )

    def _compute_polygon_vertex(angle: float) -> tuple[float, float]:
        start_point = [polygon_radius, 0]
        return _apply_rotation(start_point, angle)

    def _get_angles(n_sides: int, rotation: float) -> list[float]:
        angles = []
        degrees = 360 / n_sides
        # Start with the bottom left polygon vertex
        current_angle = (270 - 0.5 * degrees) + rotation
        for _ in range(n_sides):
            angles.append(current_angle)
            current_angle += degrees
            if current_angle > 360:
                current_angle -= 360
        return angles

    # 3. Variable Declarations
    angles = _get_angles(n_sides, rotation)

    # 4. Compute Vertices
    return [_compute_polygon_vertex(angle) for angle in angles]


def _color_diff(
    color1: float | tuple[int, ...], color2: float | tuple[int, ...]
) -> float:
    """
    Uses 1-norm distance to calculate difference between two values.
    """
    first = color1 if isinstance(color1, tuple) else (color1,)
    second = color2 if isinstance(color2, tuple) else (color2,)

    return sum(abs(first[i] - second[i]) for i in range(len(second)))
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# the Image class wrapper
#
# partial release history:
# 1995-09-09 fl   Created
# 1996-03-11 fl   PIL release 0.0 (proof of concept)
# 1996-04-30 fl   PIL release 0.1b1
# 1999-07-28 fl   PIL release 1.0 final
# 2000-06-07 fl   PIL release 1.1
# 2000-10-20 fl   PIL release 1.1.1
# 2001-05-07 fl   PIL release 1.1.2
# 2002-03-15 fl   PIL release 1.1.3
# 2003-05-10 fl   PIL release 1.1.4
# 2005-03-28 fl   PIL release 1.1.5
# 2006-12-02 fl   PIL release 1.1.6
# 2009-11-15 fl   PIL release 1.1.7
#
# Copyright (c) 1997-2009 by Secret Labs AB.  All rights reserved.
# Copyright (c) 1995-2009 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

from __future__ import annotations

import abc
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import MutableMapping
from enum import IntEnum
from typing import IO, Protocol, cast

# VERSION was removed in Pillow 6.0.0.
# PILLOW_VERSION was removed in Pillow 9.0.0.
# Use __version__ instead.
from . import (
    ExifTags,
    ImageMode,
    TiffTags,
    UnidentifiedImageError,
    __version__,
    _plugins,
)
from ._binary import i32le, o32be, o32le
from ._deprecate import deprecate
from ._util import DeferredError, is_path

ElementTree: ModuleType | None
try:
    from defusedxml import ElementTree
except ImportError:
    ElementTree = None

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType
    from typing import Any, Literal

logger = logging.getLogger(__name__)


class DecompressionBombWarning(RuntimeWarning):
    pass


class DecompressionBombError(Exception):
    pass


WARN_POSSIBLE_FORMATS: bool = False

# Limit to around a quarter gigabyte for a 24-bit (3 bpp) image
MAX_IMAGE_PIXELS: int | None = int(1024 * 1024 * 1024 // 4 // 3)


try:
    # If the _imaging C module is not present, Pillow will not load.
    # Note that other modules should not refer to _imaging directly;
    # import Image and use the Image.core variable instead.
    # Also note that Image.core is not a publicly documented interface,
    # and should be considered private and subject to change.
    from . import _imaging as core

    if __version__ != getattr(core, "PILLOW_VERSION", None):
        msg = (
            "The _imaging extension was built for another version of Pillow or PIL:\n"
            f"Core version: {getattr(core, 'PILLOW_VERSION', None)}\n"
            f"Pillow version: {__version__}"
        )
        raise ImportError(msg)

except ImportError as v:
    # Explanations for ways that we know we might have an import error
    if str(v).startswith("Module use of python"):
        # The _imaging C module is present, but not compiled for
        # the right version (windows only).  Print a warning, if
        # possible.
        warnings.warn(
            "The _imaging extension was built for another version of Python.",
            RuntimeWarning,
        )
    elif str(v).startswith("The _imaging extension"):
        warnings.warn(str(v), RuntimeWarning)
    # Fail here anyway. Don't let people run with a mostly broken Pillow.
    # see docs/porting.rst
    raise


#
# Constants


# transpose
class Transpose(IntEnum):
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4
    TRANSPOSE = 5
    TRANSVERSE = 6


# transforms (also defined in Imaging.h)
class Transform(IntEnum):
    AFFINE = 0
    EXTENT = 1
    PERSPECTIVE = 2
    QUAD = 3
    MESH = 4


# resampling filters (also defined in Imaging.h)
class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1


_filters_support = {
    Resampling.BOX: 0.5,
    Resampling.BILINEAR: 1.0,
    Resampling.HAMMING: 1.0,
    Resampling.BICUBIC: 2.0,
    Resampling.LANCZOS: 3.0,
}


# dithers
class Dither(IntEnum):
    NONE = 0
    ORDERED = 1  # Not yet implemented
    RASTERIZE = 2  # Not yet implemented
    FLOYDSTEINBERG = 3  # default


# palettes/quantizers
class Palette(IntEnum):
    WEB = 0
    ADAPTIVE = 1


class Quantize(IntEnum):
    MEDIANCUT = 0
    MAXCOVERAGE = 1
    FASTOCTREE = 2
    LIBIMAGEQUANT = 3


module = sys.modules[__name__]
for enum in (Transpose, Transform, Resampling, Dither, Palette, Quantize):
    for item in enum:
        setattr(module, item.name, item.value)


if hasattr(core, "DEFAULT_STRATEGY"):
    DEFAULT_STRATEGY = core.DEFAULT_STRATEGY
    FILTERED = core.FILTERED
    HUFFMAN_ONLY = core.HUFFMAN_ONLY
    RLE = core.RLE
    FIXED = core.FIXED


# --------------------------------------------------------------------
# Registries

TYPE_CHECKING = False
if TYPE_CHECKING:
    import mmap
    from xml.etree.ElementTree import Element

    from IPython.lib.pretty import PrettyPrinter

    from . import ImageFile, ImageFilter, ImagePalette, ImageQt, TiffImagePlugin
    from ._typing import CapsuleType, NumpyArray, StrOrBytesPath
ID: list[str] = []
OPEN: dict[
    str,
    tuple[
        Callable[[IO[bytes], str | bytes], ImageFile.ImageFile],
        Callable[[bytes], bool | str] | None,
    ],
] = {}
MIME: dict[str, str] = {}
SAVE: dict[str, Callable[[Image, IO[bytes], str | bytes], None]] = {}
SAVE_ALL: dict[str, Callable[[Image, IO[bytes], str | bytes], None]] = {}
EXTENSION: dict[str, str] = {}
DECODERS: dict[str, type[ImageFile.PyDecoder]] = {}
ENCODERS: dict[str, type[ImageFile.PyEncoder]] = {}

# --------------------------------------------------------------------
# Modes

_ENDIAN = "<" if sys.byteorder == "little" else ">"


def _conv_type_shape(im: Image) -> tuple[tuple[int, ...], str]:
    m = ImageMode.getmode(im.mode)
    shape: tuple[int, ...] = (im.height, im.width)
    extra = len(m.bands)
    if extra != 1:
        shape += (extra,)
    return shape, m.typestr


MODES = [
    "1",
    "CMYK",
    "F",
    "HSV",
    "I",
    "I;16",
    "I;16B",
    "I;16L",
    "I;16N",
    "L",
    "LA",
    "La",
    "LAB",
    "P",
    "PA",
    "RGB",
    "RGBA",
    "RGBa",
    "RGBX",
    "YCbCr",
]

# raw modes that may be memory mapped.  NOTE: if you change this, you
# may have to modify the stride calculation in map.c too!
_MAPMODES = ("L", "P", "RGBX", "RGBA", "CMYK", "I;16", "I;16L", "I;16B")


def getmodebase(mode: str) -> str:
    """
    Gets the "base" mode for given mode.  This function returns "L" for
    images that contain grayscale data, and "RGB" for images that
    contain color data.

    :param mode: Input mode.
    :returns: "L" or "RGB".
    :exception KeyError: If the input mode was not a standard mode.
    """
    return ImageMode.getmode(mode).basemode


def getmodetype(mode: str) -> str:
    """
    Gets the storage type mode.  Given a mode, this function returns a
    single-layer mode suitable for storing individual bands.

    :param mode: Input mode.
    :returns: "L", "I", or "F".
    :exception KeyError: If the input mode was not a standard mode.
    """
    return ImageMode.getmode(mode).basetype


def getmodebandnames(mode: str) -> tuple[str, ...]:
    """
    Gets a list of individual band names.  Given a mode, this function returns
    a tuple containing the names of individual bands (use
    :py:method:`~PIL.Image.getmodetype` to get the mode used to store each
    individual band.

    :param mode: Input mode.
    :returns: A tuple containing band names.  The length of the tuple
        gives the number of bands in an image of the given mode.
    :exception KeyError: If the input mode was not a standard mode.
    """
    return ImageMode.getmode(mode).bands


def getmodebands(mode: str) -> int:
    """
