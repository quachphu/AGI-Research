            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        return TextClassificationOutputElement.parse_obj_as_list(response)[0]  # type: ignore

    @overload
    async def text_generation(
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
    ) -> AsyncIterable[TextGenerationStreamOutput]: ...

    @overload
    async def text_generation(
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
    async def text_generation(
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
    ) -> AsyncIterable[str]: ...

    @overload
    async def text_generation(
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
    async def text_generation(
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
    ) -> str | TextGenerationOutput | AsyncIterable[str] | AsyncIterable[TextGenerationStreamOutput]: ...

    async def text_generation(
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
    ) -> str | TextGenerationOutput | AsyncIterable[str] | AsyncIterable[TextGenerationStreamOutput]:
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
            `Union[str, TextGenerationOutput, AsyncIterable[str], AsyncIterable[TextGenerationStreamOutput]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `AsyncIterable[str]`
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        # Case 1: generate text
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful for large generation.
        >>> async for token in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
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
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
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
        >>> async for details in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
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
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
        )

        # Case 5: generate constrained output using grammar
        >>> response = await client.text_generation(
        ...     prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
        ...     model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        ...     max_new_tokens=100,
        ...     repetition_penalty=1.3,
        ...     grammar={
        ...         "type": "json",
        ...         "value": {
        ...             "properties": {
        ...                 "location": {"type": "string"},
        ...                 "activity": {"type": "string"},
        ...                 "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...                 "animals": {"type": "array", "items": {"type": "string"}},
        ...             },
        ...             "required": ["location", "activity", "animals_seen", "animals"],
        ...         },
        ...     },
        ... )
        >>> json.loads(response)
        {
            "activity": "bike riding",
            "animals": ["puppy", "cat", "raccoon"],
            "animals_seen": 3,
            "location": "park"
        }
        ```
        """
        if decoder_input_details and not details:
            warnings.warn(
                "`decoder_input_details=True` has been passed to the server but `details=False` is set meaning that"
                " the output from the server will be truncated."
            )
            decoder_input_details = False

        # Build payload
        parameters = {
            "adapter_id": adapter_id,
            "best_of": best_of,
            "decoder_input_details": decoder_input_details,
            "details": details,
            "do_sample": do_sample,
            "frequency_penalty": frequency_penalty,
            "grammar": grammar,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "return_full_text": return_full_text,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_k": top_k,
            "top_n_tokens": top_n_tokens,
            "top_p": top_p,
            "truncate": truncate,
            "typical_p": typical_p,
            "watermark": watermark,
        }

        # Remove some parameters if not a TGI server
        unsupported_kwargs = _get_unsupported_text_generation_kwargs(model)
        if len(unsupported_kwargs) > 0:
            # The server does not support some parameters
            # => means it is not a TGI server
            # => remove unsupported parameters and warn the user

            ignored_parameters = []
            for key in unsupported_kwargs:
                if parameters.get(key):
                    ignored_parameters.append(key)
                parameters.pop(key, None)
            if len(ignored_parameters) > 0:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Ignoring following parameters:"
                    f" {', '.join(ignored_parameters)}.",
                    UserWarning,
                )
            if details:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Parameter `details=True` will"
                    " be ignored meaning only the generated text will be returned.",
                    UserWarning,
                )
                details = False
            if stream:
                raise ValueError(
                    "API endpoint/model for text-generation is not served via TGI. Cannot return output as a stream."
                    " Please pass `stream=False` as input."
                )

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-generation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters=parameters,
            extra_payload={"stream": stream},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )

        # Handle errors separately for more precise error messages
        try:
            bytes_output = await self._inner_post(request_parameters, stream=stream or False)
        except HfHubHTTPError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(str(e))
            if isinstance(e, BadRequestError) and match:
                unused_params = [kwarg.strip("' ") for kwarg in match.group(1).split(",")]
                _set_unsupported_text_generation_kwargs(model, unused_params)
                return await self.text_generation(  # type: ignore
                    prompt=prompt,
                    details=details,
                    stream=stream,
                    model=model_id,
                    adapter_id=adapter_id,
                    best_of=best_of,
                    decoder_input_details=decoder_input_details,
                    do_sample=do_sample,
                    frequency_penalty=frequency_penalty,
                    grammar=grammar,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    return_full_text=return_full_text,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_n_tokens=top_n_tokens,
                    top_p=top_p,
                    truncate=truncate,
                    typical_p=typical_p,
                    watermark=watermark,
                )
            raise_text_generation_error(e)

        # Parse output
        if stream:
            return _async_stream_text_generation_response(bytes_output, details)  # type: ignore

        data = _bytes_to_dict(bytes_output)  # type: ignore

        # Data can be a single element (dict) or an iterable of dicts where we select the first element of.
        if isinstance(data, list):
            data = data[0]
        response = provider_helper.get_response(data, request_parameters)
        return TextGenerationOutput.parse_obj_as_instance(response) if details else response["generated_text"]

    async def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        model: str | None = None,
        scheduler: str | None = None,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> "Image":
        """
        Generate an image based on a given text using a specified model.

        > [!WARNING]
        > You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        > [!TIP]
        > You can pass provider-specific parameters to the model by using the `extra_body` argument.

        Args:
            prompt (`str`):
                The prompt to generate an image from.
            negative_prompt (`str`, *optional*):
                One prompt to guide what NOT to include in image generation.
            height (`int`, *optional*):
                The height in pixels of the output image
            width (`int`, *optional*):
                The width in pixels of the output image
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                prompt, but values too high may cause saturation and other artifacts.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-image model will be used.
                Defaults to None.
            scheduler (`str`, *optional*):
                Override the scheduler with a compatible one.
            seed (`int`, *optional*):
                Seed for the random number generator.
            extra_body (`dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.

        Returns:
            `Image`: The generated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
        >>> image.save("astronaut.png")

        >>> image = await client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     negative_prompt="low resolution, blurry",
        ...     model="stabilityai/stable-diffusion-2-1",
        ... )
        >>> image.save("better_astronaut.png")
        ```
        Example using a third-party provider directly. Usage will be billed on your fal.ai account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="fal-ai",  # Use fal.ai provider
        ...     api_key="fal-ai-api-key",  # Pass your fal.ai API key
        ... )
        >>> image = client.text_to_image(
        ...     "A majestic lion in a fantasy forest",
        ...     model="black-forest-labs/FLUX.1-schnell",
        ... )
        >>> image.save("lion.png")
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> image = client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     model="black-forest-labs/FLUX.1-dev",
        ... )
        >>> image.save("astronaut.png")
        ```

        Example using Replicate provider with extra parameters
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> image = client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     model="black-forest-labs/FLUX.1-schnell",
        ...     extra_body={"output_quality": 100},
        ... )
        >>> image.save("astronaut.png")
        ```

        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-image", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters={
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return _bytes_to_image(response)

    async def text_to_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        guidance_scale: float | None = None,
        negative_prompt: list[str] | None = None,
        num_frames: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> bytes:
        """
        Generate a video based on a given text.

        > [!TIP]
        > You can pass provider-specific parameters to the model by using the `extra_body` argument.

        Args:
            prompt (`str`):
                The prompt to generate a video from.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-video model will be used.
                Defaults to None.
            guidance_scale (`float`, *optional*):
                A higher guidance scale value encourages the model to generate videos closely linked to the text
                prompt, but values too high may cause saturation and other artifacts.
            negative_prompt (`list[str]`, *optional*):
                One or several prompt to guide what NOT to include in video generation.
            num_frames (`float`, *optional*):
                The num_frames parameter determines how many video frames are generated.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            seed (`int`, *optional*):
                Seed for the random number generator.
            extra_body (`dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.

        Returns:
            `bytes`: The generated video.

        Example:

        Example using a third-party provider directly. Usage will be billed on your fal.ai account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="fal-ai",  # Using fal.ai provider
        ...     api_key="fal-ai-api-key",  # Pass your fal.ai API key
        ... )
        >>> video = client.text_to_video(
        ...     "A majestic lion running in a fantasy forest",
        ...     model="tencent/HunyuanVideo",
        ... )
        >>> with open("lion.mp4", "wb") as file:
        ...     file.write(video)
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Using replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> video = client.text_to_video(
        ...     "A cat running in a park",
        ...     model="genmo/mochi-1-preview",
        ... )
        >>> with open("cat.mp4", "wb") as file:
        ...     file.write(video)
        ```

        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-video", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters={
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return response

    async def text_to_speech(
        self,
        text: str,
        *,
        model: str | None = None,
        do_sample: bool | None = None,
        early_stopping: Union[bool, "TextToSpeechEarlyStoppingEnum"] | None = None,
        epsilon_cutoff: float | None = None,
        eta_cutoff: float | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        min_length: int | None = None,
        min_new_tokens: int | None = None,
        num_beam_groups: int | None = None,
        num_beams: int | None = None,
        penalty_alpha: float | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        typical_p: float | None = None,
        use_cache: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> bytes:
        """
        Synthesize an audio of a voice pronouncing a given text.

        > [!TIP]
        > You can pass provider-specific parameters to the model by using the `extra_body` argument.

        Args:
            text (`str`):
                The text to synthesize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-speech model will be used.
                Defaults to None.
            do_sample (`bool`, *optional*):
                Whether to use sampling instead of greedy decoding when generating new tokens.
            early_stopping (`Union[bool, "TextToSpeechEarlyStoppingEnum"]`, *optional*):
                Controls the stopping condition for beam-based methods.
            epsilon_cutoff (`float`, *optional*):
                If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
                epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on
                the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            eta_cutoff (`float`, *optional*):
                Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly
                between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff)
                * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token
                probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3,
                depending on the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            max_length (`int`, *optional*):
                The maximum length (in tokens) of the generated text, including the input.
            max_new_tokens (`int`, *optional*):
                The maximum number of tokens to generate. Takes precedence over max_length.
            min_length (`int`, *optional*):
                The minimum length (in tokens) of the generated text, including the input.
            min_new_tokens (`int`, *optional*):
                The minimum number of tokens to generate. Takes precedence over min_length.
            num_beam_groups (`int`, *optional*):
                Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.
                See [this paper](https://hf.co/papers/1610.02424) for more details.
            num_beams (`int`, *optional*):
                Number of beams to use for beam search.
            penalty_alpha (`float`, *optional*):
                The value balances the model confidence and the degeneration penalty in contrastive search decoding.
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*):
                If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
                top_p or higher are kept for generation.
            typical_p (`float`, *optional*):
                Local typicality measures how similar the conditional probability of predicting a target token next is
                to the expected conditional probability of predicting a random token next, given the partial text
                already generated. If set to float < 1, the smallest set of the most locally typical tokens with
                probabilities that add up to typical_p or higher are kept for generation. See [this
                paper](https://hf.co/papers/2202.00666) for more details.
            use_cache (`bool`, *optional*):
                Whether the model should use the past last key/values attentions to speed up decoding
            extra_body (`dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            `bytes`: The generated audio.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from pathlib import Path
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> audio = await client.text_to_speech("Hello world")
        >>> Path("hello_world.flac").write_bytes(audio)
        ```

        Example using a third-party provider directly. Usage will be billed on your Replicate account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",
        ...     api_key="your-replicate-api-key",  # Pass your Replicate API key directly
        ... )
        >>> audio = client.text_to_speech(
        ...     text="Hello world",
        ...     model="OuteAI/OuteTTS-0.3-500M",
        ... )
        >>> Path("hello_world.flac").write_bytes(audio)
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> audio =client.text_to_speech(
        ...     text="Hello world",
        ...     model="OuteAI/OuteTTS-0.3-500M",
        ... )
        >>> Path("hello_world.flac").write_bytes(audio)
        ```
        Example using Replicate provider with extra parameters
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> audio = client.text_to_speech(
        ...     "Hello, my name is Kororo, an awesome text-to-speech model.",
        ...     model="hexgrad/Kokoro-82M",
        ...     extra_body={"voice": "af_nicole"},
        ... )
        >>> Path("hello.flac").write_bytes(audio)
        ```

        Example music-gen using "YuE-s1-7B-anneal-en-cot" on fal.ai
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> lyrics = '''
        ... [verse]
        ... In the town where I was born
        ... Lived a man who sailed to sea
        ... And he told us of his life
        ... In the land of submarines
        ... So we sailed on to the sun
        ... 'Til we found a sea of green
        ... And we lived beneath the waves
        ... In our yellow submarine

        ... [chorus]
        ... We all live in a yellow submarine
        ... Yellow submarine, yellow submarine
        ... We all live in a yellow submarine
        ... Yellow submarine, yellow submarine
        ... '''
        >>> genres = "pavarotti-style tenor voice"
        >>> client = InferenceClient(
        ...     provider="fal-ai",
        ...     model="m-a-p/YuE-s1-7B-anneal-en-cot",
        ...     api_key=...,
        ... )
        >>> audio = client.text_to_speech(lyrics, extra_body={"genres": genres})
        >>> with open("output.mp3", "wb") as f:
        ...     f.write(audio)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-speech", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "do_sample": do_sample,
                "early_stopping": early_stopping,
                "epsilon_cutoff": epsilon_cutoff,
                "eta_cutoff": eta_cutoff,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
                "min_new_tokens": min_new_tokens,
                "num_beam_groups": num_beam_groups,
                "num_beams": num_beams,
                "penalty_alpha": penalty_alpha,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "typical_p": typical_p,
                "use_cache": use_cache,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response)
        return response

    async def token_classification(
        self,
        text: str,
        *,
        model: str | None = None,
        aggregation_strategy: Optional["TokenClassificationAggregationStrategy"] = None,
        ignore_labels: list[str] | None = None,
        stride: int | None = None,
    ) -> list[TokenClassificationOutputElement]:
        """
        Perform token classification on the given text.
        Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the token classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended token classification model will be used.
                Defaults to None.
            aggregation_strategy (`"TokenClassificationAggregationStrategy"`, *optional*):
                The strategy used to fuse tokens based on model predictions
            ignore_labels (`list[str`, *optional*):
                A list of labels to ignore
            stride (`int`, *optional*):
                The number of overlapping tokens between chunks when splitting the input text.

        Returns:
            `list[TokenClassificationOutputElement]`: List of [`TokenClassificationOutputElement`] items containing the entity group, confidence score, word, start and end index.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
        [
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9971321225166321,
                word='Sarah Jessica Parker',
                start=11,
                end=31,
            ),
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9773476123809814,
                word='Jessica',
                start=52,
                end=59,
            )
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="token-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "aggregation_strategy": aggregation_strategy,
                "ignore_labels": ignore_labels,
                "stride": stride,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        return TokenClassificationOutputElement.parse_obj_as_list(response)

    async def translation(
        self,
        text: str,
        *,
        model: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        clean_up_tokenization_spaces: bool | None = None,
        truncation: Optional["TranslationTruncationStrategy"] = None,
        generate_parameters: dict[str, Any] | None = None,
    ) -> TranslationOutput:
        """
        Convert text from one language to another.

        Check out https://huggingface.co/tasks/translation for more information on how to choose the best model for
        your specific use case. Source and target languages usually depend on the model.
        However, it is possible to specify source and target languages for certain models. If you are working with one of these models,
        you can use `src_lang` and `tgt_lang` arguments to pass the relevant information.

        Args:
            text (`str`):
                A string to be translated.
            model (`str`, *optional*):
                The model to use for the translation task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended translation model will be used.
                Defaults to None.
            src_lang (`str`, *optional*):
                The source language of the text. Required for models that can translate from multiple languages.
            tgt_lang (`str`, *optional*):
                Target language to translate to. Required for models that can translate to multiple languages.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            truncation (`"TranslationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
            generate_parameters (`dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.

        Returns:
            [`TranslationOutput`]: The generated translated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If only one of the `src_lang` and `tgt_lang` arguments are provided.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.translation("My name is Wolfgang and I live in Berlin")
        'Mein Name ist Wolfgang und ich lebe in Berlin.'
        >>> await client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
        TranslationOutput(translation_text='Je m'appelle Wolfgang et je vis à Berlin.')
        ```

        Specifying languages:
        ```py
        >>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        "Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
        ```
        """
        # Throw error if only one of `src_lang` and `tgt_lang` was given
        if src_lang is not None and tgt_lang is None:
            raise ValueError("You cannot specify `src_lang` without specifying `tgt_lang`.")

        if src_lang is None and tgt_lang is not None:
            raise ValueError("You cannot specify `tgt_lang` without specifying `src_lang`.")

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="translation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
                "truncation": truncation,
                "generate_parameters": generate_parameters,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        return TranslationOutput.parse_obj_as_list(response)[0]

    async def visual_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
    ) -> list[VisualQuestionAnsweringOutputElement]:
        """
        Answering open-ended questions based on an image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image for the context. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the visual question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended visual question answering model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.
        Returns:
            `list[VisualQuestionAnsweringOutputElement]`: a list of [`VisualQuestionAnsweringOutputElement`] items containing the predicted label and associated probability.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.visual_question_answering(
        ...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
        ...     question="What is the animal doing?"
        ... )
        [
            VisualQuestionAnsweringOutputElement(score=0.778609573841095, answer='laying down'),
            VisualQuestionAnsweringOutputElement(score=0.6957435607910156, answer='sitting'),
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="visual-question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
            extra_payload={"question": question, "image": _b64_encode(image)},
        )
        response = await self._inner_post(request_parameters)
        return VisualQuestionAnsweringOutputElement.parse_obj_as_list(response)

    async def zero_shot_classification(
        self,
        text: str,
        candidate_labels: list[str],
        *,
        multi_label: bool | None = False,
        hypothesis_template: str | None = None,
        model: str | None = None,
    ) -> list[ZeroShotClassificationOutputElement]:
        """
        Provide as input a text and a set of candidate labels to classify the input text.

        Args:
            text (`str`):
                The input text to classify.
            candidate_labels (`list[str]`):
                The set of possible class labels to classify the text into.
            multi_label (`bool`, *optional*):
                Whether multiple candidate labels can be true. If false, the scores are normalized such that the sum of
                the label likelihoods for each sequence is 1. If true, the labels are considered independent and
                probabilities are normalized for each candidate.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the text classification by
                replacing the placeholder with the candidate labels.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot classification model will be used.


        Returns:
            `list[ZeroShotClassificationOutputElement]`: List of [`ZeroShotClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example with `multi_label=False`:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> text = (
        ...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
        ...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
        ...     " mysteries when he went for a run up a hill in Nice, France."
        ... )
        >>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
        >>> await client.zero_shot_classification(text, labels)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
            ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
            ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
        ]
        >>> await client.zero_shot_classification(text, labels, multi_label=True)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.9829297661781311),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.755190908908844),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.0005462635890580714),
            ZeroShotClassificationOutputElement(label='archeology', score=0.00047131875180639327),
            ZeroShotClassificationOutputElement(label='robots', score=0.00030448526376858354),
        ]
        ```

        Example with `multi_label=True` and a custom `hypothesis_template`:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.zero_shot_classification(
        ...    text="I really like our dinner and I'm very happy. I don't like the weather though.",
        ...    candidate_labels=["positive", "negative", "pessimistic", "optimistic"],
        ...    multi_label=True,
        ...    hypothesis_template="This text is {} towards the weather"
        ... )
        [
            ZeroShotClassificationOutputElement(label='negative', score=0.9231801629066467),
            ZeroShotClassificationOutputElement(label='pessimistic', score=0.8760990500450134),
            ZeroShotClassificationOutputElement(label='optimistic', score=0.0008674879791215062),
            ZeroShotClassificationOutputElement(label='positive', score=0.0005250611575320363)
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="zero-shot-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "candidate_labels": candidate_labels,
                "multi_label": multi_label,
                "hypothesis_template": hypothesis_template,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        output = _bytes_to_dict(response)
        return ZeroShotClassificationOutputElement.parse_obj_as_list(output)

    async def zero_shot_image_classification(
        self,
        image: ContentT,
        candidate_labels: list[str],
        *,
        model: str | None = None,
        hypothesis_template: str | None = None,
    ) -> list[ZeroShotImageClassificationOutputElement]:
        """
        Provide input image and text labels to predict text labels for the image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image to caption. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            candidate_labels (`list[str]`):
                The candidate labels for this image
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot image classification model will be used.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the image classification by
                replacing the placeholder with the candidate labels.

        Returns:
            `list[ZeroShotImageClassificationOutputElement]`: List of [`ZeroShotImageClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            [`HfHubHTTPError`]:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> await client.zero_shot_image_classification(
        ...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        ...     candidate_labels=["dog", "cat", "horse"],
        ... )
        [ZeroShotImageClassificationOutputElement(label='dog', score=0.956),...]
        ```
        """
        # Raise ValueError if input is less than 2 labels
        if len(candidate_labels) < 2:
            raise ValueError("You must specify at least 2 classes to compare.")

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="zero-shot-image-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "candidate_labels": candidate_labels,
                "hypothesis_template": hypothesis_template,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = await self._inner_post(request_parameters)
        return ZeroShotImageClassificationOutputElement.parse_obj_as_list(response)

    async def get_endpoint_info(self, *, model: str | None = None) -> dict[str, Any]:
        """
        Get information about the deployed endpoint.

        This endpoint is only available on endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).
        Endpoints powered by `transformers` return an empty payload.

        Args:
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `dict[str, Any]`: Information about the endpoint.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> await client.get_endpoint_info()
        {
            'model_id': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'model_sha': None,
            'model_dtype': 'torch.float16',
            'model_device_type': 'cuda',
            'model_pipeline_tag': None,
            'max_concurrent_requests': 128,
            'max_best_of': 2,
            'max_stop_sequences': 4,
            'max_input_length': 8191,
            'max_total_tokens': 8192,
            'waiting_served_ratio': 0.3,
            'max_batch_total_tokens': 1259392,
            'max_waiting_tokens': 20,
            'max_batch_size': None,
            'validation_workers': 32,
            'max_client_batch_size': 4,
            'version': '2.0.2',
            'sha': 'dccab72549635c7eb5ddb17f43f0b7cdff07c214',
            'docker_label': 'sha-dccab72'
        }
        ```
        """
        if self.provider != "hf-inference":
            raise ValueError(f"Getting endpoint info is not supported on '{self.provider}'.")

        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if model.startswith(("http://", "https://")):
            url = model.rstrip("/") + "/info"
        else:
            url = f"{constants.INFERENCE_ENDPOINT}/models/{model}/info"

        client = await self._get_async_client()
        response = await client.get(url, headers=build_hf_headers(token=self.token))
        hf_raise_for_status(response)
        return response.json()

    async def health_check(self, model: str | None = None) -> bool:
        """
        Check the health of the deployed endpoint.

        Health check is only available with Inference Endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).

        Args:
            model (`str`, *optional*):
                URL of the Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `bool`: True if everything is working fine.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
        >>> await client.health_check()
        True
        ```
        """
        if self.provider != "hf-inference":
            raise ValueError(f"Health check is not supported on '{self.provider}'.")

        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if not model.startswith(("http://", "https://")):
            raise ValueError("Model must be an Inference Endpoint URL.")
        url = model.rstrip("/") + "/health"

        client = await self._get_async_client()
        response = await client.get(url, headers=build_hf_headers(token=self.token))
        return response.status_code == 200

    @property
    def chat(self) -> "ProxyClientChat":
        return ProxyClientChat(self)


class _ProxyClient:
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    def __init__(self, client: AsyncInferenceClient):
        self._client = client


class ProxyClientChat(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def completions(self) -> "ProxyClientChatCompletions":
        return ProxyClientChatCompletions(self._client)


class ProxyClientChatCompletions(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def create(self):
        return self._client.chat_completion
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ImageToImageTargetSize(BaseInferenceType):
    """The size in pixels of the output image. This parameter is only supported by some
    providers and for specific models. It will be ignored when unsupported.
    """

    height: int
    width: int


@dataclass_with_extra
class ImageToImageParameters(BaseInferenceType):
    """Additional inference parameters for Image To Image"""

    guidance_scale: float | None = None
    """For diffusion models. A higher guidance scale value encourages the model to generate
    images closely linked to the text prompt at the expense of lower image quality.
    """
    negative_prompt: str | None = None
    """One prompt to guide what NOT to include in image generation."""
    num_inference_steps: int | None = None
    """For diffusion models. The number of denoising steps. More denoising steps usually lead to
    a higher quality image at the expense of slower inference.
    """
    prompt: str | None = None
    """The text prompt to guide the image generation."""
    target_size: ImageToImageTargetSize | None = None
    """The size in pixels of the output image. This parameter is only supported by some
    providers and for specific models. It will be ignored when unsupported.
    """


@dataclass_with_extra
class ImageToImageInput(BaseInferenceType):
    """Inputs for Image To Image inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: ImageToImageParameters | None = None
    """Additional inference parameters for Image To Image"""


@dataclass_with_extra
class ImageToImageOutput(BaseInferenceType):
    """Outputs of inference for the Image To Image task"""

    image: Any
    """The output image returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class QuestionAnsweringInputData(BaseInferenceType):
    """One (context, question) pair to answer"""

    context: str
    """The context to be used for answering the question"""
    question: str
    """The question to be answered"""


@dataclass_with_extra
class QuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters for Question Answering"""

    align_to_words: bool | None = None
    """Attempts to align the answer to real words. Improves quality on space separated
    languages. Might hurt on non-space-separated languages (like Japanese or Chinese)
    """
    doc_stride: int | None = None
    """If the context is too long to fit with the question for the model, it will be split in
    several chunks with some overlap. This argument controls the size of that overlap.
    """
    handle_impossible_answer: bool | None = None
    """Whether to accept impossible as an answer."""
    max_answer_len: int | None = None
    """The maximum length of predicted answers (e.g., only answers with a shorter length are
    considered).
    """
    max_question_len: int | None = None
    """The maximum length of the question after tokenization. It will be truncated if needed."""
    max_seq_len: int | None = None
    """The maximum length of the total sentence (context + question) in tokens of each chunk
    passed to the model. The context will be split in several chunks (using docStride as
    overlap) if needed.
    """
    top_k: int | None = None
    """The number of answers to return (will be chosen by order of likelihood). Note that we
    return less than topk answers if there are not enough options available within the
    context.
    """


@dataclass_with_extra
class QuestionAnsweringInput(BaseInferenceType):
    """Inputs for Question Answering inference"""

    inputs: QuestionAnsweringInputData
    """One (context, question) pair to answer"""
    parameters: QuestionAnsweringParameters | None = None
    """Additional inference parameters for Question Answering"""


@dataclass_with_extra
class QuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Question Answering task"""

    answer: str
    """The answer to the question."""
    end: int
    """The character position in the input where the answer ends."""
    score: float
    """The probability associated to the answer."""
    start: int
    """The character position in the input where the answer begins."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal

from .base import BaseInferenceType, dataclass_with_extra


TypeEnum = Literal["json", "regex", "json_schema"]


@dataclass_with_extra
class TextGenerationInputGrammarType(BaseInferenceType):
    type: "TypeEnum"
    value: Any
    """A string that represents a [JSON Schema](https://json-schema.org/).
    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions.
    """


@dataclass_with_extra
class TextGenerationInputGenerateParameters(BaseInferenceType):
    adapter_id: str | None = None
    """Lora adapter id"""
    best_of: int | None = None
    """Generate best_of sequences and return the one if the highest token logprobs."""
    decoder_input_details: bool | None = None
    """Whether to return decoder input token logprobs and ids."""
    details: bool | None = None
    """Whether to return generation details."""
    do_sample: bool | None = None
    """Activate logits sampling."""
    frequency_penalty: float | None = None
    """The parameter for frequency penalty. 1.0 means no penalty
    Penalize new tokens based on their existing frequency in the text so far,
    decreasing the model's likelihood to repeat the same line verbatim.
    """
    grammar: TextGenerationInputGrammarType | None = None
    max_new_tokens: int | None = None
    """Maximum number of tokens to generate."""
    repetition_penalty: float | None = None
    """The parameter for repetition penalty. 1.0 means no penalty.
    See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """
    return_full_text: bool | None = None
    """Whether to prepend the prompt to the generated text"""
    seed: int | None = None
    """Random sampling seed."""
    stop: list[str] | None = None
    """Stop generating tokens if a member of `stop` is generated."""
    temperature: float | None = None
    """The value used to module the logits distribution."""
    top_k: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_n_tokens: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-n-filtering."""
    top_p: float | None = None
    """Top-p value for nucleus sampling."""
    truncate: int | None = None
    """Truncate inputs tokens to the given size."""
    typical_p: float | None = None
    """Typical Decoding mass
    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)
    for more information.
    """
    watermark: bool | None = None
    """Watermarking with [A Watermark for Large Language
    Models](https://arxiv.org/abs/2301.10226).
    """


@dataclass_with_extra
class TextGenerationInput(BaseInferenceType):
    """Text Generation Input.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    inputs: str
    parameters: TextGenerationInputGenerateParameters | None = None
    stream: bool | None = None


TextGenerationOutputFinishReason = Literal["length", "eos_token", "stop_sequence"]


@dataclass_with_extra
class TextGenerationOutputPrefillToken(BaseInferenceType):
    id: int
    logprob: float
    text: str


@dataclass_with_extra
class TextGenerationOutputToken(BaseInferenceType):
    id: int
    logprob: float
    special: bool
    text: str


@dataclass_with_extra
class TextGenerationOutputBestOfSequence(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_text: str
    generated_tokens: int
    prefill: list[TextGenerationOutputPrefillToken]
    tokens: list[TextGenerationOutputToken]
    seed: int | None = None
    top_tokens: list[list[TextGenerationOutputToken]] | None = None


@dataclass_with_extra
class TextGenerationOutputDetails(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_tokens: int
    prefill: list[TextGenerationOutputPrefillToken]
    tokens: list[TextGenerationOutputToken]
    best_of_sequences: list[TextGenerationOutputBestOfSequence] | None = None
    seed: int | None = None
    top_tokens: list[list[TextGenerationOutputToken]] | None = None


@dataclass_with_extra
class TextGenerationOutput(BaseInferenceType):
    """Text Generation Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    generated_text: str
    details: TextGenerationOutputDetails | None = None


@dataclass_with_extra
class TextGenerationStreamOutputStreamDetails(BaseInferenceType):
    finish_reason: "TextGenerationOutputFinishReason"
    generated_tokens: int
    input_length: int
    seed: int | None = None


@dataclass_with_extra
class TextGenerationStreamOutputToken(BaseInferenceType):
    id: int
    logprob: float
    special: bool
    text: str


@dataclass_with_extra
class TextGenerationStreamOutput(BaseInferenceType):
    """Text Generation Stream Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    index: int
    token: TextGenerationStreamOutputToken
    details: TextGenerationStreamOutputStreamDetails | None = None
    generated_text: str | None = None
    top_tokens: list[TextGenerationStreamOutputToken] | None = None
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class AudioToAudioInput(BaseInferenceType):
    """Inputs for Audio to Audio inference"""

    inputs: Any
    """The input audio data"""


@dataclass_with_extra
class AudioToAudioOutputElement(BaseInferenceType):
    """Outputs of inference for the Audio To Audio task
    A generated audio file with its label.
    """

    blob: Any
    """The generated audio file."""
    content_type: str
    """The content type of audio file."""
    label: str
    """The label of the audio file."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ImageToVideoTargetSize(BaseInferenceType):
    """The size in pixel of the output video frames."""

    height: int
    width: int


@dataclass_with_extra
class ImageToVideoParameters(BaseInferenceType):
    """Additional inference parameters for Image To Video"""

    guidance_scale: float | None = None
    """For diffusion models. A higher guidance scale value encourages the model to generate
    videos closely linked to the text prompt at the expense of lower image quality.
    """
    negative_prompt: str | None = None
    """One prompt to guide what NOT to include in video generation."""
    num_frames: float | None = None
    """The num_frames parameter determines how many video frames are generated."""
    num_inference_steps: int | None = None
    """The number of denoising steps. More denoising steps usually lead to a higher quality
    video at the expense of slower inference.
    """
    prompt: str | None = None
    """The text prompt to guide the video generation."""
    seed: int | None = None
    """Seed for the random number generator."""
    target_size: ImageToVideoTargetSize | None = None
    """The size in pixel of the output video frames."""


@dataclass_with_extra
class ImageToVideoInput(BaseInferenceType):
    """Inputs for Image To Video inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: ImageToVideoParameters | None = None
    """Additional inference parameters for Image To Video"""


@dataclass_with_extra
class ImageToVideoOutput(BaseInferenceType):
    """Outputs of inference for the Image To Video task"""

    video: Any
    """The generated video returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


AudioClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass_with_extra
class AudioClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Audio Classification"""

    function_to_apply: Optional["AudioClassificationOutputTransform"] = None
    """The function to apply to the model outputs in order to retrieve the scores."""
    top_k: int | None = None
    """When specified, limits the output to the top K most probable classes."""


@dataclass_with_extra
class AudioClassificationInput(BaseInferenceType):
    """Inputs for Audio Classification inference"""

    inputs: str
    """The input audio data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the audio data as a raw bytes payload.
    """
    parameters: AudioClassificationParameters | None = None
    """Additional inference parameters for Audio Classification"""


@dataclass_with_extra
class AudioClassificationOutputElement(BaseInferenceType):
    """Outputs for Audio Classification inference"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class VisualQuestionAnsweringInputData(BaseInferenceType):
    """One (image, question) pair to answer"""

    image: Any
    """The image."""
    question: str
    """The question to answer based on the image."""


@dataclass_with_extra
class VisualQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters for Visual Question Answering"""

    top_k: int | None = None
    """The number of answers to return (will be chosen by order of likelihood). Note that we
    return less than topk answers if there are not enough options available within the
    context.
    """


@dataclass_with_extra
class VisualQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Visual Question Answering inference"""

    inputs: VisualQuestionAnsweringInputData
    """One (image, question) pair to answer"""
    parameters: VisualQuestionAnsweringParameters | None = None
    """Additional inference parameters for Visual Question Answering"""


@dataclass_with_extra
class VisualQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Visual Question Answering task"""

    score: float
    """The associated score / probability"""
    answer: str | None = None
    """The answer to the question"""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Union

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ChatCompletionInputURL(BaseInferenceType):
    url: str


ChatCompletionInputMessageChunkType = Literal["text", "image_url"]


@dataclass_with_extra
class ChatCompletionInputMessageChunk(BaseInferenceType):
    type: "ChatCompletionInputMessageChunkType"
    image_url: ChatCompletionInputURL | None = None
    text: str | None = None


@dataclass_with_extra
class ChatCompletionInputFunctionDefinition(BaseInferenceType):
    name: str
    parameters: Any
    description: str | None = None


@dataclass_with_extra
class ChatCompletionInputToolCall(BaseInferenceType):
    function: ChatCompletionInputFunctionDefinition
    id: str
    type: str


@dataclass_with_extra
class ChatCompletionInputMessage(BaseInferenceType):
    role: str
    content: list[ChatCompletionInputMessageChunk] | str | None = None
    name: str | None = None
    tool_calls: list[ChatCompletionInputToolCall] | None = None


@dataclass_with_extra
class ChatCompletionInputJSONSchema(BaseInferenceType):
    name: str
    """
    The name of the response format.
    """
    description: str | None = None
    """
    A description of what the response format is for, used by the model to determine
    how to respond in the format.
    """
    schema: dict[str, object] | None = None
    """
    The schema for the response format, described as a JSON Schema object. Learn how
    to build JSON schemas [here](https://json-schema.org/).
    """
    strict: bool | None = None
    """
    Whether to enable strict schema adherence when generating the output. If set to
    true, the model will always follow the exact schema defined in the `schema`
    field.
    """


@dataclass_with_extra
class ChatCompletionInputResponseFormatText(BaseInferenceType):
    type: Literal["text"]


@dataclass_with_extra
class ChatCompletionInputResponseFormatJSONSchema(BaseInferenceType):
    type: Literal["json_schema"]
    json_schema: ChatCompletionInputJSONSchema


@dataclass_with_extra
class ChatCompletionInputResponseFormatJSONObject(BaseInferenceType):
    type: Literal["json_object"]


ChatCompletionInputGrammarType = Union[
    ChatCompletionInputResponseFormatText,
    ChatCompletionInputResponseFormatJSONSchema,
    ChatCompletionInputResponseFormatJSONObject,
]


@dataclass_with_extra
class ChatCompletionInputStreamOptions(BaseInferenceType):
    include_usage: bool | None = None
    """If set, an additional chunk will be streamed before the data: [DONE] message. The usage
    field on this chunk shows the token usage statistics for the entire request, and the
    choices field will always be an empty array. All other chunks will also include a usage
    field, but with a null value.
    """


@dataclass_with_extra
class ChatCompletionInputFunctionName(BaseInferenceType):
    name: str


@dataclass_with_extra
class ChatCompletionInputToolChoiceClass(BaseInferenceType):
    function: ChatCompletionInputFunctionName


ChatCompletionInputToolChoiceEnum = Literal["auto", "none", "required"]


@dataclass_with_extra
class ChatCompletionInputTool(BaseInferenceType):
    function: ChatCompletionInputFunctionDefinition
    type: str


@dataclass_with_extra
class ChatCompletionInput(BaseInferenceType):
    """Chat Completion Input.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    messages: list[ChatCompletionInputMessage]
    """A list of messages comprising the conversation so far."""
    frequency_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    frequency in the text so far,
    decreasing the model's likelihood to repeat the same line verbatim.
    """
    logit_bias: list[float] | None = None
    """UNUSED
    Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON
    object that maps tokens
    (specified by their token ID in the tokenizer) to an associated bias value from -100 to
    100. Mathematically,
    the bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model,
    but values between -1 and 1 should decrease or increase likelihood of selection; values
    like -100 or 100 should
    result in a ban or exclusive selection of the relevant token.
    """
    logprobs: bool | None = None
    """Whether to return log probabilities of the output tokens or not. If true, returns the log
    probabilities of each
    output token returned in the content of message.
    """
    max_tokens: int | None = None
    """The maximum number of tokens that can be generated in the chat completion."""
    model: str | None = None
    """[UNUSED] ID of the model to use. See the model endpoint compatibility table for details
    on which models work with the Chat API.
    """
    n: int | None = None
    """UNUSED
    How many chat completion choices to generate for each input message. Note that you will
    be charged based on the
    number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
    """
    presence_penalty: float | None = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    appear in the text so far,
    increasing the model's likelihood to talk about new topics
    """
    response_format: ChatCompletionInputGrammarType | None = None
    seed: int | None = None
    stop: list[str] | None = None
    """Up to 4 sequences where the API will stop generating further tokens."""
    stream: bool | None = None
    stream_options: ChatCompletionInputStreamOptions | None = None
    temperature: float | None = None
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    output more random, while
    lower values like 0.2 will make it more focused and deterministic.
    We generally recommend altering this or `top_p` but not both.
    """
    tool_choice: Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"] | None = None
    tool_prompt: str | None = None
    """A prompt to be appended before the tools"""
    tools: list[ChatCompletionInputTool] | None = None
    """A list of tools the model may call. Currently, only functions are supported as a tool.
    Use this to provide a list of
    functions the model may generate JSON inputs for.
    """
    top_logprobs: int | None = None
    """An integer between 0 and 5 specifying the number of most likely tokens to return at each
    token position, each with
    an associated log probability. logprobs must be set to true if this parameter is used.
    """
    top_p: float | None = None
    """An alternative to sampling with temperature, called nucleus sampling, where the model
    considers the results of the
    tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%
    probability mass are considered.
    """


@dataclass_with_extra
class ChatCompletionOutputTopLogprob(BaseInferenceType):
    logprob: float
    token: str


@dataclass_with_extra
class ChatCompletionOutputLogprob(BaseInferenceType):
    logprob: float
    token: str
    top_logprobs: list[ChatCompletionOutputTopLogprob]


@dataclass_with_extra
class ChatCompletionOutputLogprobs(BaseInferenceType):
    content: list[ChatCompletionOutputLogprob]


@dataclass_with_extra
class ChatCompletionOutputFunctionDefinition(BaseInferenceType):
    arguments: str
    name: str
    description: str | None = None


@dataclass_with_extra
class ChatCompletionOutputToolCall(BaseInferenceType):
    function: ChatCompletionOutputFunctionDefinition
    id: str
    type: str


@dataclass_with_extra
class ChatCompletionOutputMessage(BaseInferenceType):
    role: str
    content: str | None = None
    reasoning: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatCompletionOutputToolCall] | None = None


@dataclass_with_extra
class ChatCompletionOutputComplete(BaseInferenceType):
    finish_reason: str
    index: int
    message: ChatCompletionOutputMessage
    logprobs: ChatCompletionOutputLogprobs | None = None


@dataclass_with_extra
class ChatCompletionOutputUsage(BaseInferenceType):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass_with_extra
class ChatCompletionOutput(BaseInferenceType):
    """Chat Completion Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    choices: list[ChatCompletionOutputComplete]
    created: int
    id: str
    model: str
    system_fingerprint: str
    usage: ChatCompletionOutputUsage


@dataclass_with_extra
class ChatCompletionStreamOutputFunction(BaseInferenceType):
    arguments: str
    name: str | None = None


@dataclass_with_extra
class ChatCompletionStreamOutputDeltaToolCall(BaseInferenceType):
    function: ChatCompletionStreamOutputFunction
    id: str
    index: int
    type: str


@dataclass_with_extra
class ChatCompletionStreamOutputDelta(BaseInferenceType):
    role: str
    content: str | None = None
    reasoning: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatCompletionStreamOutputDeltaToolCall] | None = None


@dataclass_with_extra
class ChatCompletionStreamOutputTopLogprob(BaseInferenceType):
    logprob: float
    token: str


@dataclass_with_extra
class ChatCompletionStreamOutputLogprob(BaseInferenceType):
    logprob: float
    token: str
    top_logprobs: list[ChatCompletionStreamOutputTopLogprob]


@dataclass_with_extra
class ChatCompletionStreamOutputLogprobs(BaseInferenceType):
    content: list[ChatCompletionStreamOutputLogprob]


@dataclass_with_extra
class ChatCompletionStreamOutputChoice(BaseInferenceType):
    delta: ChatCompletionStreamOutputDelta
    index: int
    finish_reason: str | None = None
    logprobs: ChatCompletionStreamOutputLogprobs | None = None


@dataclass_with_extra
class ChatCompletionStreamOutputUsage(BaseInferenceType):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass_with_extra
class ChatCompletionStreamOutput(BaseInferenceType):
    """Chat Completion Stream Output.
    Auto-generated from TGI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tgi-import.ts.
    """

    choices: list[ChatCompletionStreamOutputChoice]
    created: int
    id: str
    model: str
    system_fingerprint: str
    usage: ChatCompletionStreamOutputUsage | None = None
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class DocumentQuestionAnsweringInputData(BaseInferenceType):
    """One (document, question) pair to answer"""

    image: Any
    """The image on which the question is asked"""
    question: str
    """A question to ask of the document"""


@dataclass_with_extra
class DocumentQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters for Document Question Answering"""

    doc_stride: int | None = None
    """If the words in the document are too long to fit with the question for the model, it will
    be split in several chunks with some overlap. This argument controls the size of that
    overlap.
    """
    handle_impossible_answer: bool | None = None
    """Whether to accept impossible as an answer"""
    lang: str | None = None
    """Language to use while running OCR. Defaults to english."""
    max_answer_len: int | None = None
    """The maximum length of predicted answers (e.g., only answers with a shorter length are
    considered).
    """
    max_question_len: int | None = None
    """The maximum length of the question after tokenization. It will be truncated if needed."""
    max_seq_len: int | None = None
    """The maximum length of the total sentence (context + question) in tokens of each chunk
    passed to the model. The context will be split in several chunks (using doc_stride as
    overlap) if needed.
    """
    top_k: int | None = None
    """The number of answers to return (will be chosen by order of likelihood). Can return less
    than top_k answers if there are not enough options available within the context.
    """
    word_boxes: list[list[float] | str] | None = None
    """A list of words and bounding boxes (normalized 0->1000). If provided, the inference will
    skip the OCR step and use the provided bounding boxes instead.
    """


@dataclass_with_extra
class DocumentQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Document Question Answering inference"""

    inputs: DocumentQuestionAnsweringInputData
    """One (document, question) pair to answer"""
    parameters: DocumentQuestionAnsweringParameters | None = None
    """Additional inference parameters for Document Question Answering"""


@dataclass_with_extra
class DocumentQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Document Question Answering task"""

    answer: str
    """The answer to the question."""
    end: int
    """The end word index of the answer (in the OCR’d version of the input or provided word
    boxes).
    """
    score: float
    """The probability associated to the answer."""
    start: int
    """The start word index of the answer (in the OCR’d version of the input or provided word
    boxes).
    """
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class TextToVideoParameters(BaseInferenceType):
    """Additional inference parameters for Text To Video"""

    guidance_scale: float | None = None
    """A higher guidance scale value encourages the model to generate videos closely linked to
    the text prompt, but values too high may cause saturation and other artifacts.
    """
    negative_prompt: list[str] | None = None
    """One or several prompt to guide what NOT to include in video generation."""
    num_frames: float | None = None
    """The num_frames parameter determines how many video frames are generated."""
    num_inference_steps: int | None = None
    """The number of denoising steps. More denoising steps usually lead to a higher quality
    video at the expense of slower inference.
    """
    seed: int | None = None
    """Seed for the random number generator."""


@dataclass_with_extra
class TextToVideoInput(BaseInferenceType):
    """Inputs for Text To Video inference"""

    inputs: str
    """The input text data (sometimes called "prompt")"""
    parameters: TextToVideoParameters | None = None
    """Additional inference parameters for Text To Video"""


@dataclass_with_extra
class TextToVideoOutput(BaseInferenceType):
    """Outputs of inference for the Text To Video task"""

    video: Any
    """The generated video returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ImageTextToVideoTargetSize(BaseInferenceType):
    """The size in pixel of the output video frames."""

    height: int
    width: int


@dataclass_with_extra
class ImageTextToVideoParameters(BaseInferenceType):
    """Additional inference parameters for Image Text To Video"""

    guidance_scale: float | None = None
    """For diffusion models. A higher guidance scale value encourages the model to generate
    videos closely linked to the text prompt at the expense of lower image quality.
    """
    negative_prompt: str | None = None
    """One prompt to guide what NOT to include in video generation."""
    num_frames: float | None = None
    """The num_frames parameter determines how many video frames are generated."""
    num_inference_steps: int | None = None
    """The number of denoising steps. More denoising steps usually lead to a higher quality
    video at the expense of slower inference.
    """
    prompt: str | None = None
    """The text prompt to guide the video generation. Either this or inputs (image) must be
    provided.
    """
    seed: int | None = None
    """Seed for the random number generator."""
    target_size: ImageTextToVideoTargetSize | None = None
    """The size in pixel of the output video frames."""


@dataclass_with_extra
class ImageTextToVideoInput(BaseInferenceType):
    """Inputs for Image Text To Video inference. Either inputs (image) or prompt (in parameters)
    must be provided, or both.
    """

    inputs: str | None = None
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload. Either this or prompt must be
    provided.
    """
    parameters: ImageTextToVideoParameters | None = None
    """Additional inference parameters for Image Text To Video"""


@dataclass_with_extra
class ImageTextToVideoOutput(BaseInferenceType):
    """Outputs of inference for the Image Text To Video task"""

    video: Any
    """The generated video returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Union

from .base import BaseInferenceType, dataclass_with_extra


ImageToTextEarlyStoppingEnum = Literal["never"]


@dataclass_with_extra
class ImageToTextGenerationParameters(BaseInferenceType):
    """Parametrization of the text generation process"""

    do_sample: bool | None = None
    """Whether to use sampling instead of greedy decoding when generating new tokens."""
    early_stopping: Union[bool, "ImageToTextEarlyStoppingEnum"] | None = None
    """Controls the stopping condition for beam-based methods."""
    epsilon_cutoff: float | None = None
    """If set to float strictly between 0 and 1, only tokens with a conditional probability
    greater than epsilon_cutoff will be sampled. In the paper, suggested values range from
    3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language
    Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    """
    eta_cutoff: float | None = None
    """Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to
    float strictly between 0 and 1, a token is only considered if it is greater than either
    eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter
    term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In
    the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
    See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)
    for more details.
    """
    max_length: int | None = None
    """The maximum length (in tokens) of the generated text, including the input."""
    max_new_tokens: int | None = None
    """The maximum number of tokens to generate. Takes precedence over max_length."""
    min_length: int | None = None
    """The minimum length (in tokens) of the generated text, including the input."""
    min_new_tokens: int | None = None
    """The minimum number of tokens to generate. Takes precedence over min_length."""
    num_beam_groups: int | None = None
    """Number of groups to divide num_beams into in order to ensure diversity among different
    groups of beams. See [this paper](https://hf.co/papers/1610.02424) for more details.
    """
    num_beams: int | None = None
    """Number of beams to use for beam search."""
    penalty_alpha: float | None = None
    """The value balances the model confidence and the degeneration penalty in contrastive
    search decoding.
    """
    temperature: float | None = None
    """The value used to modulate the next token probabilities."""
    top_k: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: float | None = None
    """If set to float < 1, only the smallest set of most probable tokens with probabilities
    that add up to top_p or higher are kept for generation.
    """
    typical_p: float | None = None
    """Local typicality measures how similar the conditional probability of predicting a target
    token next is to the expected conditional probability of predicting a random token next,
    given the partial text already generated. If set to float < 1, the smallest set of the
    most locally typical tokens with probabilities that add up to typical_p or higher are
    kept for generation. See [this paper](https://hf.co/papers/2202.00666) for more details.
    """
    use_cache: bool | None = None
    """Whether the model should use the past last key/values attentions to speed up decoding"""


@dataclass_with_extra
class ImageToTextParameters(BaseInferenceType):
    """Additional inference parameters for Image To Text"""

    generation_parameters: ImageToTextGenerationParameters | None = None
    """Parametrization of the text generation process"""
    max_new_tokens: int | None = None
    """The amount of maximum tokens to generate."""


@dataclass_with_extra
class ImageToTextInput(BaseInferenceType):
    """Inputs for Image To Text inference"""

    inputs: Any
    """The input image data"""
    parameters: ImageToTextParameters | None = None
    """Additional inference parameters for Image To Text"""


@dataclass_with_extra
class ImageToTextOutput(BaseInferenceType):
    """Outputs of inference for the Image To Text task"""

    generated_text: Any
    image_to_text_output_generated_text: str | None = None
    """The generated text."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class DepthEstimationInput(BaseInferenceType):
    """Inputs for Depth Estimation inference"""

    inputs: Any
    """The input image data"""
    parameters: dict[str, Any] | None = None
    """Additional inference parameters for Depth Estimation"""


@dataclass_with_extra
class DepthEstimationOutput(BaseInferenceType):
    """Outputs of inference for the Depth Estimation task"""

    depth: Any
    """The predicted depth as an image"""
    predicted_depth: Any
    """The predicted depth as a tensor"""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


TokenClassificationAggregationStrategy = Literal["none", "simple", "first", "average", "max"]


@dataclass_with_extra
class TokenClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Token Classification"""

    aggregation_strategy: Optional["TokenClassificationAggregationStrategy"] = None
    """The strategy used to fuse tokens based on model predictions"""
    ignore_labels: list[str] | None = None
    """A list of labels to ignore"""
    stride: int | None = None
    """The number of overlapping tokens between chunks when splitting the input text."""


@dataclass_with_extra
class TokenClassificationInput(BaseInferenceType):
    """Inputs for Token Classification inference"""

    inputs: str
    """The input text data"""
    parameters: TokenClassificationParameters | None = None
    """Additional inference parameters for Token Classification"""


@dataclass_with_extra
class TokenClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Token Classification task"""

    end: int
    """The character position in the input where this group ends."""
    score: float
    """The associated score / probability"""
    start: int
    """The character position in the input where this group begins."""
    word: str
    """The corresponding text"""
    entity: str | None = None
    """The predicted label for a single token"""
    entity_group: str | None = None
    """The predicted label for a group of one or more tokens"""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Union

from .base import BaseInferenceType, dataclass_with_extra


AutomaticSpeechRecognitionEarlyStoppingEnum = Literal["never"]


@dataclass_with_extra
class AutomaticSpeechRecognitionGenerationParameters(BaseInferenceType):
    """Parametrization of the text generation process"""

    do_sample: bool | None = None
    """Whether to use sampling instead of greedy decoding when generating new tokens."""
    early_stopping: Union[bool, "AutomaticSpeechRecognitionEarlyStoppingEnum"] | None = None
    """Controls the stopping condition for beam-based methods."""
    epsilon_cutoff: float | None = None
    """If set to float strictly between 0 and 1, only tokens with a conditional probability
    greater than epsilon_cutoff will be sampled. In the paper, suggested values range from
    3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language
    Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    """
    eta_cutoff: float | None = None
    """Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to
    float strictly between 0 and 1, a token is only considered if it is greater than either
    eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter
    term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In
    the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
    See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)
    for more details.
    """
    max_length: int | None = None
    """The maximum length (in tokens) of the generated text, including the input."""
    max_new_tokens: int | None = None
    """The maximum number of tokens to generate. Takes precedence over max_length."""
    min_length: int | None = None
    """The minimum length (in tokens) of the generated text, including the input."""
    min_new_tokens: int | None = None
    """The minimum number of tokens to generate. Takes precedence over min_length."""
    num_beam_groups: int | None = None
    """Number of groups to divide num_beams into in order to ensure diversity among different
    groups of beams. See [this paper](https://hf.co/papers/1610.02424) for more details.
    """
    num_beams: int | None = None
    """Number of beams to use for beam search."""
    penalty_alpha: float | None = None
    """The value balances the model confidence and the degeneration penalty in contrastive
    search decoding.
    """
    temperature: float | None = None
    """The value used to modulate the next token probabilities."""
    top_k: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: float | None = None
    """If set to float < 1, only the smallest set of most probable tokens with probabilities
    that add up to top_p or higher are kept for generation.
    """
    typical_p: float | None = None
    """Local typicality measures how similar the conditional probability of predicting a target
    token next is to the expected conditional probability of predicting a random token next,
    given the partial text already generated. If set to float < 1, the smallest set of the
    most locally typical tokens with probabilities that add up to typical_p or higher are
    kept for generation. See [this paper](https://hf.co/papers/2202.00666) for more details.
    """
    use_cache: bool | None = None
    """Whether the model should use the past last key/values attentions to speed up decoding"""


@dataclass_with_extra
class AutomaticSpeechRecognitionParameters(BaseInferenceType):
    """Additional inference parameters for Automatic Speech Recognition"""

    generation_parameters: AutomaticSpeechRecognitionGenerationParameters | None = None
    """Parametrization of the text generation process"""
    return_timestamps: bool | None = None
    """Whether to output corresponding timestamps with the generated text"""


@dataclass_with_extra
class AutomaticSpeechRecognitionInput(BaseInferenceType):
    """Inputs for Automatic Speech Recognition inference"""

    inputs: str
    """The input audio data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the audio data as a raw bytes payload.
    """
    parameters: AutomaticSpeechRecognitionParameters | None = None
    """Additional inference parameters for Automatic Speech Recognition"""


@dataclass_with_extra
class AutomaticSpeechRecognitionOutputChunk(BaseInferenceType):
    text: str
    """A chunk of text identified by the model"""
    timestamp: list[float]
    """The start and end timestamps corresponding with the text"""


@dataclass_with_extra
class AutomaticSpeechRecognitionOutput(BaseInferenceType):
    """Outputs of inference for the Automatic Speech Recognition task"""

    text: str
    """The recognized text."""
    chunks: list[AutomaticSpeechRecognitionOutputChunk] | None = None
    """When returnTimestamps is enabled, chunks contains a list of audio chunks identified by
    the model.
    """
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Union

from .base import BaseInferenceType, dataclass_with_extra


TextToAudioEarlyStoppingEnum = Literal["never"]


@dataclass_with_extra
class TextToAudioGenerationParameters(BaseInferenceType):
    """Parametrization of the text generation process"""

    do_sample: bool | None = None
    """Whether to use sampling instead of greedy decoding when generating new tokens."""
    early_stopping: Union[bool, "TextToAudioEarlyStoppingEnum"] | None = None
    """Controls the stopping condition for beam-based methods."""
    epsilon_cutoff: float | None = None
    """If set to float strictly between 0 and 1, only tokens with a conditional probability
    greater than epsilon_cutoff will be sampled. In the paper, suggested values range from
    3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language
    Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    """
    eta_cutoff: float | None = None
    """Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to
    float strictly between 0 and 1, a token is only considered if it is greater than either
    eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter
    term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In
    the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
    See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)
    for more details.
    """
    max_length: int | None = None
    """The maximum length (in tokens) of the generated text, including the input."""
    max_new_tokens: int | None = None
    """The maximum number of tokens to generate. Takes precedence over max_length."""
    min_length: int | None = None
    """The minimum length (in tokens) of the generated text, including the input."""
    min_new_tokens: int | None = None
    """The minimum number of tokens to generate. Takes precedence over min_length."""
    num_beam_groups: int | None = None
    """Number of groups to divide num_beams into in order to ensure diversity among different
    groups of beams. See [this paper](https://hf.co/papers/1610.02424) for more details.
    """
    num_beams: int | None = None
    """Number of beams to use for beam search."""
    penalty_alpha: float | None = None
    """The value balances the model confidence and the degeneration penalty in contrastive
    search decoding.
    """
    temperature: float | None = None
    """The value used to modulate the next token probabilities."""
    top_k: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: float | None = None
    """If set to float < 1, only the smallest set of most probable tokens with probabilities
    that add up to top_p or higher are kept for generation.
    """
    typical_p: float | None = None
    """Local typicality measures how similar the conditional probability of predicting a target
    token next is to the expected conditional probability of predicting a random token next,
    given the partial text already generated. If set to float < 1, the smallest set of the
    most locally typical tokens with probabilities that add up to typical_p or higher are
    kept for generation. See [this paper](https://hf.co/papers/2202.00666) for more details.
    """
    use_cache: bool | None = None
    """Whether the model should use the past last key/values attentions to speed up decoding"""


@dataclass_with_extra
class TextToAudioParameters(BaseInferenceType):
    """Additional inference parameters for Text To Audio"""

    generation_parameters: TextToAudioGenerationParameters | None = None
    """Parametrization of the text generation process"""


@dataclass_with_extra
class TextToAudioInput(BaseInferenceType):
    """Inputs for Text To Audio inference"""

    inputs: str
    """The input text data"""
    parameters: TextToAudioParameters | None = None
    """Additional inference parameters for Text To Audio"""


@dataclass_with_extra
class TextToAudioOutput(BaseInferenceType):
    """Outputs of inference for the Text To Audio task"""

    audio: Any
    """The generated audio waveform."""
    sampling_rate: float
    """The sampling rate of the generated audio waveform."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ZeroShotClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Zero Shot Classification"""

    candidate_labels: list[str]
    """The set of possible class labels to classify the text into."""
    hypothesis_template: str | None = None
    """The sentence used in conjunction with `candidate_labels` to attempt the text
    classification by replacing the placeholder with the candidate labels.
    """
    multi_label: bool | None = None
    """Whether multiple candidate labels can be true. If false, the scores are normalized such
    that the sum of the label likelihoods for each sequence is 1. If true, the labels are
    considered independent and probabilities are normalized for each candidate.
    """


@dataclass_with_extra
class ZeroShotClassificationInput(BaseInferenceType):
    """Inputs for Zero Shot Classification inference"""

    inputs: str
    """The text to classify"""
    parameters: ZeroShotClassificationParameters
    """Additional inference parameters for Zero Shot Classification"""


@dataclass_with_extra
class ZeroShotClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class SentenceSimilarityInputData(BaseInferenceType):
    sentences: list[str]
    """A list of strings which will be compared against the source_sentence."""
    source_sentence: str
    """The string that you wish to compare the other strings with. This can be a phrase,
    sentence, or longer passage, depending on the model being used.
    """


@dataclass_with_extra
class SentenceSimilarityInput(BaseInferenceType):
    """Inputs for Sentence similarity inference"""

    inputs: SentenceSimilarityInputData
    parameters: dict[str, Any] | None = None
    """Additional inference parameters for Sentence Similarity"""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


ImageSegmentationSubtask = Literal["instance", "panoptic", "semantic"]


@dataclass_with_extra
class ImageSegmentationParameters(BaseInferenceType):
    """Additional inference parameters for Image Segmentation"""

    mask_threshold: float | None = None
    """Threshold to use when turning the predicted masks into binary values."""
    overlap_mask_area_threshold: float | None = None
    """Mask overlap threshold to eliminate small, disconnected segments."""
    subtask: Optional["ImageSegmentationSubtask"] = None
    """Segmentation task to be performed, depending on model capabilities."""
    threshold: float | None = None
    """Probability threshold to filter out predicted masks."""


@dataclass_with_extra
class ImageSegmentationInput(BaseInferenceType):
    """Inputs for Image Segmentation inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: ImageSegmentationParameters | None = None
    """Additional inference parameters for Image Segmentation"""


@dataclass_with_extra
class ImageSegmentationOutputElement(BaseInferenceType):
    """Outputs of inference for the Image Segmentation task
    A predicted mask / segment
    """

    label: str
    """The label of the predicted segment."""
    mask: str
    """The corresponding mask as a black-and-white image (base64-encoded)."""
    score: float | None = None
    """The score or confidence degree the model has."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


Text2TextGenerationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


@dataclass_with_extra
class Text2TextGenerationParameters(BaseInferenceType):
    """Additional inference parameters for Text2text Generation"""

    clean_up_tokenization_spaces: bool | None = None
    """Whether to clean up the potential extra spaces in the text output."""
    generate_parameters: dict[str, Any] | None = None
    """Additional parametrization of the text generation algorithm"""
    truncation: Optional["Text2TextGenerationTruncationStrategy"] = None
    """The truncation strategy to use"""


@dataclass_with_extra
class Text2TextGenerationInput(BaseInferenceType):
    """Inputs for Text2text Generation inference"""

    inputs: str
    """The input text data"""
    parameters: Text2TextGenerationParameters | None = None
    """Additional inference parameters for Text2text Generation"""


@dataclass_with_extra
class Text2TextGenerationOutput(BaseInferenceType):
    """Outputs of inference for the Text2text Generation task"""

    generated_text: Any
    text2_text_generation_output_generated_text: str | None = None
    """The generated text."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


SummarizationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


@dataclass_with_extra
class SummarizationParameters(BaseInferenceType):
    """Additional inference parameters for summarization."""

    clean_up_tokenization_spaces: bool | None = None
    """Whether to clean up the potential extra spaces in the text output."""
    generate_parameters: dict[str, Any] | None = None
    """Additional parametrization of the text generation algorithm."""
    truncation: Optional["SummarizationTruncationStrategy"] = None
    """The truncation strategy to use."""


@dataclass_with_extra
class SummarizationInput(BaseInferenceType):
    """Inputs for Summarization inference"""

    inputs: str
    """The input text to summarize."""
    parameters: SummarizationParameters | None = None
    """Additional inference parameters for summarization."""


@dataclass_with_extra
class SummarizationOutput(BaseInferenceType):
    """Outputs of inference for the Summarization task"""

    summary_text: str
    """The summarized text."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


FeatureExtractionInputTruncationDirection = Literal["left", "right"]


@dataclass_with_extra
class FeatureExtractionInput(BaseInferenceType):
    """Feature Extraction Input.
    Auto-generated from TEI specs.
    For more details, check out
    https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-tei-import.ts.
    """

    inputs: list[str] | str
    """The text or list of texts to embed."""
    normalize: bool | None = None
    prompt_name: str | None = None
    """The name of the prompt that should be used by for encoding. If not set, no prompt
    will be applied.
    Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ",
    ...},
    then the sentence "What is the capital of France?" will be encoded as
    "query: What is the capital of France?" because the prompt text will be prepended before
    any text to encode.
    """
    truncate: bool | None = None
    truncation_direction: Optional["FeatureExtractionInputTruncationDirection"] = None
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ObjectDetectionParameters(BaseInferenceType):
    """Additional inference parameters for Object Detection"""

    threshold: float | None = None
    """The probability necessary to make a prediction."""


@dataclass_with_extra
class ObjectDetectionInput(BaseInferenceType):
    """Inputs for Object Detection inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: ObjectDetectionParameters | None = None
    """Additional inference parameters for Object Detection"""


@dataclass_with_extra
class ObjectDetectionBoundingBox(BaseInferenceType):
    """The predicted bounding box. Coordinates are relative to the top left corner of the input
    image.
    """

    xmax: int
    """The x-coordinate of the bottom-right corner of the bounding box."""
    xmin: int
    """The x-coordinate of the top-left corner of the bounding box."""
    ymax: int
    """The y-coordinate of the bottom-right corner of the bounding box."""
    ymin: int
    """The y-coordinate of the top-left corner of the bounding box."""


@dataclass_with_extra
class ObjectDetectionOutputElement(BaseInferenceType):
    """Outputs of inference for the Object Detection task"""

    box: ObjectDetectionBoundingBox
    """The predicted bounding box. Coordinates are relative to the top left corner of the input
    image.
    """
    label: str
    """The predicted label for the bounding box."""
    score: float
    """The associated score / probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class TableQuestionAnsweringInputData(BaseInferenceType):
    """One (table, question) pair to answer"""

    question: str
    """The question to be answered about the table"""
    table: dict[str, list[str]]
    """The table to serve as context for the questions"""


Padding = Literal["do_not_pad", "longest", "max_length"]


@dataclass_with_extra
class TableQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters for Table Question Answering"""

    padding: Optional["Padding"] = None
    """Activates and controls padding."""
    sequential: bool | None = None
    """Whether to do inference sequentially or as a batch. Batching is faster, but models like
    SQA require the inference to be done sequentially to extract relations within sequences,
    given their conversational nature.
    """
    truncation: bool | None = None
    """Activates and controls truncation."""


@dataclass_with_extra
class TableQuestionAnsweringInput(BaseInferenceType):
    """Inputs for Table Question Answering inference"""

    inputs: TableQuestionAnsweringInputData
    """One (table, question) pair to answer"""
    parameters: TableQuestionAnsweringParameters | None = None
    """Additional inference parameters for Table Question Answering"""


@dataclass_with_extra
class TableQuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Table Question Answering task"""

    answer: str
    """The answer of the question given the table. If there is an aggregator, the answer will be
    preceded by `AGGREGATOR >`.
    """
    cells: list[str]
    """list of strings made up of the answer cell values."""
    coordinates: list[list[int]]
    """Coordinates of the cells of the answers."""
    aggregator: str | None = None
    """If the model has an aggregator, this returns the aggregator."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


TranslationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


@dataclass_with_extra
class TranslationParameters(BaseInferenceType):
    """Additional inference parameters for Translation"""

    clean_up_tokenization_spaces: bool | None = None
    """Whether to clean up the potential extra spaces in the text output."""
    generate_parameters: dict[str, Any] | None = None
    """Additional parametrization of the text generation algorithm."""
    src_lang: str | None = None
    """The source language of the text. Required for models that can translate from multiple
    languages.
    """
    tgt_lang: str | None = None
    """Target language to translate to. Required for models that can translate to multiple
    languages.
    """
    truncation: Optional["TranslationTruncationStrategy"] = None
    """The truncation strategy to use."""


@dataclass_with_extra
class TranslationInput(BaseInferenceType):
    """Inputs for Translation inference"""

    inputs: str
    """The text to translate."""
    parameters: TranslationParameters | None = None
    """Additional inference parameters for Translation"""


@dataclass_with_extra
class TranslationOutput(BaseInferenceType):
    """Outputs of inference for the Translation task"""

    translation_text: str
    """The translated text."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ZeroShotObjectDetectionParameters(BaseInferenceType):
    """Additional inference parameters for Zero Shot Object Detection"""

    candidate_labels: list[str]
    """The candidate labels for this image"""


@dataclass_with_extra
class ZeroShotObjectDetectionInput(BaseInferenceType):
    """Inputs for Zero Shot Object Detection inference"""

    inputs: str
    """The input image data as a base64-encoded string."""
    parameters: ZeroShotObjectDetectionParameters
    """Additional inference parameters for Zero Shot Object Detection"""


@dataclass_with_extra
class ZeroShotObjectDetectionBoundingBox(BaseInferenceType):
    """The predicted bounding box. Coordinates are relative to the top left corner of the input
    image.
    """

    xmax: int
    xmin: int
    ymax: int
    ymin: int


@dataclass_with_extra
class ZeroShotObjectDetectionOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Object Detection task"""

    box: ZeroShotObjectDetectionBoundingBox
    """The predicted bounding box. Coordinates are relative to the top left corner of the input
    image.
    """
    label: str
    """A candidate label"""
    score: float
    """The associated score / probability"""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Union

from .base import BaseInferenceType, dataclass_with_extra


TextToSpeechEarlyStoppingEnum = Literal["never"]


@dataclass_with_extra
class TextToSpeechGenerationParameters(BaseInferenceType):
    """Parametrization of the text generation process"""

    do_sample: bool | None = None
    """Whether to use sampling instead of greedy decoding when generating new tokens."""
    early_stopping: Union[bool, "TextToSpeechEarlyStoppingEnum"] | None = None
    """Controls the stopping condition for beam-based methods."""
    epsilon_cutoff: float | None = None
    """If set to float strictly between 0 and 1, only tokens with a conditional probability
    greater than epsilon_cutoff will be sampled. In the paper, suggested values range from
    3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language
    Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    """
    eta_cutoff: float | None = None
    """Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to
    float strictly between 0 and 1, a token is only considered if it is greater than either
    eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter
    term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In
    the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
    See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)
    for more details.
    """
    max_length: int | None = None
    """The maximum length (in tokens) of the generated text, including the input."""
    max_new_tokens: int | None = None
    """The maximum number of tokens to generate. Takes precedence over max_length."""
    min_length: int | None = None
    """The minimum length (in tokens) of the generated text, including the input."""
    min_new_tokens: int | None = None
    """The minimum number of tokens to generate. Takes precedence over min_length."""
    num_beam_groups: int | None = None
    """Number of groups to divide num_beams into in order to ensure diversity among different
    groups of beams. See [this paper](https://hf.co/papers/1610.02424) for more details.
    """
    num_beams: int | None = None
    """Number of beams to use for beam search."""
    penalty_alpha: float | None = None
    """The value balances the model confidence and the degeneration penalty in contrastive
    search decoding.
    """
    temperature: float | None = None
    """The value used to modulate the next token probabilities."""
    top_k: int | None = None
    """The number of highest probability vocabulary tokens to keep for top-k-filtering."""
    top_p: float | None = None
    """If set to float < 1, only the smallest set of most probable tokens with probabilities
    that add up to top_p or higher are kept for generation.
    """
    typical_p: float | None = None
    """Local typicality measures how similar the conditional probability of predicting a target
    token next is to the expected conditional probability of predicting a random token next,
    given the partial text already generated. If set to float < 1, the smallest set of the
    most locally typical tokens with probabilities that add up to typical_p or higher are
    kept for generation. See [this paper](https://hf.co/papers/2202.00666) for more details.
    """
    use_cache: bool | None = None
    """Whether the model should use the past last key/values attentions to speed up decoding"""


@dataclass_with_extra
class TextToSpeechParameters(BaseInferenceType):
    """Additional inference parameters for Text To Speech"""

    generation_parameters: TextToSpeechGenerationParameters | None = None
    """Parametrization of the text generation process"""


@dataclass_with_extra
class TextToSpeechInput(BaseInferenceType):
    """Inputs for Text To Speech inference"""

    inputs: str
    """The input text data"""
    parameters: TextToSpeechParameters | None = None
    """Additional inference parameters for Text To Speech"""


@dataclass_with_extra
class TextToSpeechOutput(BaseInferenceType):
    """Outputs of inference for the Text To Speech task"""

    audio: Any
    """The generated audio"""
    sampling_rate: float | None = None
    """The sampling rate of the generated audio waveform."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any, Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


VideoClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass_with_extra
class VideoClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Video Classification"""

    frame_sampling_rate: int | None = None
    """The sampling rate used to select frames from the video."""
    function_to_apply: Optional["VideoClassificationOutputTransform"] = None
    """The function to apply to the model outputs in order to retrieve the scores."""
    num_frames: int | None = None
    """The number of sampled frames to consider for classification."""
    top_k: int | None = None
    """When specified, limits the output to the top K most probable classes."""


@dataclass_with_extra
class VideoClassificationInput(BaseInferenceType):
    """Inputs for Video Classification inference"""

    inputs: Any
    """The input video data"""
    parameters: VideoClassificationParameters | None = None
    """Additional inference parameters for Video Classification"""


@dataclass_with_extra
class VideoClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Video Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class TextToImageParameters(BaseInferenceType):
    """Additional inference parameters for Text To Image"""

    guidance_scale: float | None = None
    """A higher guidance scale value encourages the model to generate images closely linked to
    the text prompt, but values too high may cause saturation and other artifacts.
    """
    height: int | None = None
    """The height in pixels of the output image"""
    negative_prompt: str | None = None
    """One prompt to guide what NOT to include in image generation."""
    num_inference_steps: int | None = None
    """The number of denoising steps. More denoising steps usually lead to a higher quality
    image at the expense of slower inference.
    """
    scheduler: str | None = None
    """Override the scheduler with a compatible one."""
    seed: int | None = None
    """Seed for the random number generator."""
    width: int | None = None
    """The width in pixels of the output image"""


@dataclass_with_extra
class TextToImageInput(BaseInferenceType):
    """Inputs for Text To Image inference"""

    inputs: str
    """The input text data (sometimes called "prompt")"""
    parameters: TextToImageParameters | None = None
    """Additional inference parameters for Text To Image"""


@dataclass_with_extra
class TextToImageOutput(BaseInferenceType):
    """Outputs of inference for the Text To Image task"""

    image: Any
    """The generated image returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


TextClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass_with_extra
class TextClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Text Classification"""

    function_to_apply: Optional["TextClassificationOutputTransform"] = None
    """The function to apply to the model outputs in order to retrieve the scores."""
    top_k: int | None = None
    """When specified, limits the output to the top K most probable classes."""


@dataclass_with_extra
class TextClassificationInput(BaseInferenceType):
    """Inputs for Text Classification inference"""

    inputs: str
    """The text to classify"""
    parameters: TextClassificationParameters | None = None
    """Additional inference parameters for Text Classification"""


@dataclass_with_extra
class TextClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Text Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ImageTextToImageTargetSize(BaseInferenceType):
    """The size in pixels of the output image. This parameter is only supported by some
    providers and for specific models. It will be ignored when unsupported.
    """

    height: int
    width: int


@dataclass_with_extra
class ImageTextToImageParameters(BaseInferenceType):
    """Additional inference parameters for Image Text To Image"""

    guidance_scale: float | None = None
    """For diffusion models. A higher guidance scale value encourages the model to generate
    images closely linked to the text prompt at the expense of lower image quality.
    """
    negative_prompt: str | None = None
    """One prompt to guide what NOT to include in image generation."""
    num_inference_steps: int | None = None
    """For diffusion models. The number of denoising steps. More denoising steps usually lead to
    a higher quality image at the expense of slower inference.
    """
    prompt: str | None = None
    """The text prompt to guide the image generation. Either this or inputs (image) must be
    provided.
    """
    seed: int | None = None
    """Seed for the random number generator."""
    target_size: ImageTextToImageTargetSize | None = None
    """The size in pixels of the output image. This parameter is only supported by some
    providers and for specific models. It will be ignored when unsupported.
    """


@dataclass_with_extra
class ImageTextToImageInput(BaseInferenceType):
    """Inputs for Image Text To Image inference. Either inputs (image) or prompt (in parameters)
    must be provided, or both.
    """

    inputs: str | None = None
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload. Either this or prompt must be
    provided.
    """
    parameters: ImageTextToImageParameters | None = None
    """Additional inference parameters for Image Text To Image"""


@dataclass_with_extra
class ImageTextToImageOutput(BaseInferenceType):
    """Outputs of inference for the Image Text To Image task"""

    image: Any
    """The generated image returned as raw bytes in the payload."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Any

from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class FillMaskParameters(BaseInferenceType):
    """Additional inference parameters for Fill Mask"""

    targets: list[str] | None = None
    """When passed, the model will limit the scores to the passed targets instead of looking up
    in the whole vocabulary. If the provided targets are not in the model vocab, they will be
    tokenized and the first resulting token will be used (with a warning, and that might be
    slower).
    """
    top_k: int | None = None
    """When passed, overrides the number of predictions to return."""


@dataclass_with_extra
class FillMaskInput(BaseInferenceType):
    """Inputs for Fill Mask inference"""

    inputs: str
    """The text with masked tokens"""
    parameters: FillMaskParameters | None = None
    """Additional inference parameters for Fill Mask"""


@dataclass_with_extra
class FillMaskOutputElement(BaseInferenceType):
    """Outputs of inference for the Fill Mask task"""

    score: float
    """The corresponding probability"""
    sequence: str
    """The corresponding input with the mask token prediction."""
    token: int
    """The predicted token id (to replace the masked one)."""
    token_str: Any
    fill_mask_output_token_str: str | None = None
    """The predicted token (to replace the masked one)."""
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
"""Contains a base class for all inference types."""

import inspect
import json
import types
from dataclasses import asdict, dataclass
from typing import Any, TypeVar, get_args

from typing_extensions import dataclass_transform


T = TypeVar("T", bound="BaseInferenceType")


def _repr_with_extra(self):
    fields = list(self.__dataclass_fields__.keys())
    other_fields = list(k for k in self.__dict__ if k not in fields)
    return f"{self.__class__.__name__}({', '.join(f'{k}={self.__dict__[k]!r}' for k in fields + other_fields)})"


@dataclass_transform()
def dataclass_with_extra(cls: type[T]) -> type[T]:
    """Decorator to add a custom __repr__ method to a dataclass, showing all fields, including extra ones.

    This decorator only works with dataclasses that inherit from `BaseInferenceType`.
    """
    cls = dataclass(cls)
    cls.__repr__ = _repr_with_extra  # type: ignore[method-assign]
    return cls


@dataclass
class BaseInferenceType(dict):
    """Base class for all inference types.

    Object is a dataclass and a dict for backward compatibility but plan is to remove the dict part in the future.

    Handle parsing from dict, list and json strings in a permissive way to ensure future-compatibility (e.g. all fields
    are made optional, and non-expected fields are added as dict attributes).
    """

    @classmethod
    def parse_obj_as_list(cls: type[T], data: bytes | str | list | dict) -> list[T]:
        """Alias to parse server response and return a single instance.

        See `parse_obj` for more details.
        """
        output = cls.parse_obj(data)
        if not isinstance(output, list):
            raise ValueError(f"Invalid input data for {cls}. Expected a list, but got {type(output)}.")
        return output

    @classmethod
    def parse_obj_as_instance(cls: type[T], data: bytes | str | list | dict) -> T:
        """Alias to parse server response and return a single instance.

        See `parse_obj` for more details.
        """
        output = cls.parse_obj(data)
        if isinstance(output, list):
            raise ValueError(f"Invalid input data for {cls}. Expected a single instance, but got a list.")
        return output

    @classmethod
    def parse_obj(cls: type[T], data: bytes | str | list | dict) -> list[T] | T:
        """Parse server response as a dataclass or list of dataclasses.

        To enable future-compatibility, we want to handle cases where the server return more fields than expected.
        In such cases, we don't want to raise an error but still create the dataclass object. Remaining fields are
        added as dict attributes.
        """
        # Parse server response (from bytes)
        if isinstance(data, bytes):
            data = data.decode()
        if isinstance(data, str):
            data = json.loads(data)

        # If a list, parse each item individually
        if isinstance(data, list):
            return [cls.parse_obj(d) for d in data]  # type: ignore

        # At this point, we expect a dict
        if not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        init_values = {}
        other_values = {}
        for key, value in data.items():
            key = normalize_key(key)
            if key in cls.__dataclass_fields__ and cls.__dataclass_fields__[key].init:
                if isinstance(value, dict) or isinstance(value, list):
                    field_type = cls.__dataclass_fields__[key].type

                    # if `field_type` is a `BaseInferenceType`, parse it
                    if inspect.isclass(field_type) and issubclass(field_type, BaseInferenceType):
                        value = field_type.parse_obj(value)

                    # otherwise, recursively parse nested dataclasses (if possible)
                    # `get_args` returns handle Union and Optional for us
                    else:
                        expected_types = get_args(field_type)
                        for expected_type in expected_types:
                            if (
                                isinstance(expected_type, types.GenericAlias) and expected_type.__origin__ is list
                            ) or getattr(expected_type, "_name", None) == "List":
                                expected_type = get_args(expected_type)[
                                    0
                                ]  # assume same type for all items in the list
                            if inspect.isclass(expected_type) and issubclass(expected_type, BaseInferenceType):
                                value = expected_type.parse_obj(value)
                                break
                init_values[key] = value
            else:
                other_values[key] = value

        # Make all missing fields default to None
        # => ensure that dataclass initialization will never fail even if the server does not return all fields.
        for key in cls.__dataclass_fields__:
            if key not in init_values:
                init_values[key] = None

        # Initialize dataclass with expected values
        item = cls(**init_values)

        # Add remaining fields as dict attributes
        item.update(other_values)

        # Add remaining fields as extra dataclass fields.
        # They won't be part of the dataclass fields but will be accessible as attributes.
        # Use @dataclass_with_extra to show them in __repr__.
        item.__dict__.update(other_values)
        return item

    def __post_init__(self):
        self.update(asdict(self))

    def __setitem__(self, __key: Any, __value: Any) -> None:
        # Hacky way to keep dataclass values in sync when dict is updated
        super().__setitem__(__key, __value)
        if __key in self.__dataclass_fields__ and getattr(self, __key, None) != __value:
            self.__setattr__(__key, __value)
        return

    def __setattr__(self, __name: str, __value: Any) -> None:
        # Hacky way to keep dict values is sync when dataclass is updated
        super().__setattr__(__name, __value)
        if self.get(__name) != __value:
            self[__name] = __value
        return


def normalize_key(key: str) -> str:
    # e.g "content-type" -> "content_type", "Accept" -> "accept"
    return key.replace("-", "_").replace(" ", "_").lower()
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from .base import BaseInferenceType, dataclass_with_extra


@dataclass_with_extra
class ZeroShotImageClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Zero Shot Image Classification"""

    candidate_labels: list[str]
    """The candidate labels for this image"""
    hypothesis_template: str | None = None
    """The sentence used in conjunction with `candidate_labels` to attempt the image
    classification by replacing the placeholder with the candidate labels.
    """


@dataclass_with_extra
class ZeroShotImageClassificationInput(BaseInferenceType):
    """Inputs for Zero Shot Image Classification inference"""

    inputs: str
    """The input image data to classify as a base64-encoded string."""
    parameters: ZeroShotImageClassificationParameters
    """Additional inference parameters for Zero Shot Image Classification"""


@dataclass_with_extra
class ZeroShotImageClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Image Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|># Inference code generated from the JSON schema spec in @huggingface/tasks.
#
# See:
#   - script: https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/scripts/inference-codegen.ts
#   - specs:  https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks.
from typing import Literal, Optional

from .base import BaseInferenceType, dataclass_with_extra


ImageClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]


@dataclass_with_extra
class ImageClassificationParameters(BaseInferenceType):
    """Additional inference parameters for Image Classification"""

    function_to_apply: Optional["ImageClassificationOutputTransform"] = None
    """The function to apply to the model outputs in order to retrieve the scores."""
    top_k: int | None = None
    """When specified, limits the output to the top K most probable classes."""


@dataclass_with_extra
class ImageClassificationInput(BaseInferenceType):
    """Inputs for Image Classification inference"""

    inputs: str
    """The input image data as a base64-encoded string. If no `parameters` are provided, you can
    also provide the image data as a raw bytes payload.
    """
    parameters: ImageClassificationParameters | None = None
    """Additional inference parameters for Image Classification"""


@dataclass_with_extra
class ImageClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Image Classification task"""

    label: str
    """The predicted class label."""
    score: float
    """The corresponding probability."""
<|endoftext|>from ._common import BaseConversationalTask


class GroqConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="groq", base_url="https://api.groq.com")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/openai/v1/chat/completions"
<|endoftext|>import time
from abc import ABC
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import BaseConversationalTask, TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session


_PROVIDER = "zai-org"
_BASE_URL = "https://api.z.ai"
_POLLING_INTERVAL = 5  # seconds
_MAX_POLL_ATTEMPTS = 60


class ZaiTask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        headers["Accept-Language"] = "en-US,en"
        headers["x-source-channel"] = "hugging_face"
        return headers


class ZaiConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        headers["Accept-Language"] = "en-US,en"
        headers["x-source-channel"] = "hugging_face"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/api/paas/v4/chat/completions"


class ZaiTextToImageTask(ZaiTask):
    """Text-to-image task for ZAI provider using async API."""

    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/api/paas/v4/async/images/generations"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        width = parameters.pop("width", None)
        height = parameters.pop("height", None)
        size = None
        if width is not None and height is not None:
            size = f"{width}x{height}"

        payload: dict[str, Any] = {
            "model": provider_mapping_info.provider_id,
            "prompt": inputs,
        }
        if size is not None:
            payload["size"] = size

        payload.update(filter_none(parameters))
        return payload

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        """Handle async response by polling for results."""
        response_dict = _as_dict(response)

        task_id = response_dict.get("id")
        if task_id is None:
            raise ValueError("No task_id in response from ZAI API")

        task_status = response_dict.get("task_status")
        if task_status == "FAIL":
            raise ValueError(f"ZAI image generation failed for request {task_id}")

        if task_status == "PROCESSING" and request_params is not None:
            return self._poll_for_result(task_id, request_params)

        return self._extract_image(response_dict)

    def _poll_for_result(self, task_id: str, request_params: RequestParameters) -> bytes:
        """Poll the async-result endpoint until completion."""
        session = get_session()
        base_url = request_params.url.rsplit("/api/paas/v4/async/images/generations", 1)[0]
        poll_url = f"{base_url}/api/paas/v4/async-result/{task_id}"

        for _ in range(_MAX_POLL_ATTEMPTS):
            poll_response = session.get(poll_url, headers=request_params.headers)
            poll_response.raise_for_status()
            result = poll_response.json()

            task_status = result.get("task_status")
            if task_status == "SUCCESS":
                return self._extract_image(result)
            elif task_status == "FAIL":
                raise ValueError(f"Zai text-to-image generation failed for request {task_id}")

            time.sleep(_POLLING_INTERVAL)

        raise ValueError(
            f"Timed out while waiting for the result from Zai API - aborting after {_MAX_POLL_ATTEMPTS} attempts"
        )

    def _extract_image(self, result: dict) -> bytes:
        """Extract and download the image from the result."""
        image_result = result.get("image_result")
        if not image_result or not isinstance(image_result, list) or len(image_result) == 0:
            raise ValueError("No image_result in response from ZAI API")

        image_url = image_result[0].get("url")
        if not image_url:
            raise ValueError("No image URL in response from ZAI API")

        session = get_session()
        image_response = session.get(image_url)
        image_response.raise_for_status()
        return image_response.content
<|endoftext|>from ._common import BaseConversationalTask


class CerebrasConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="cerebras", base_url="https://api.cerebras.ai")
<|endoftext|>import base64
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict

from ._common import BaseConversationalTask, TaskProviderHelper, filter_none


class NscaleConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="nscale", base_url="https://inference.api.nscale.com")


class NscaleTextToImageTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="nscale", base_url="https://inference.api.nscale.com", task="text-to-image")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/images/generations"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        # Combine all parameters except inputs and parameters
        parameters = filter_none(parameters)
        if "width" in parameters and "height" in parameters:
            parameters["size"] = f"{parameters.pop('width')}x{parameters.pop('height')}"
        if "num_inference_steps" in parameters:
            parameters.pop("num_inference_steps")
        if "cfg_scale" in parameters:
            parameters.pop("cfg_scale")
        payload = {
            "response_format": "b64_json",
            "prompt": inputs,
            "model": mapped_model,
            **parameters,
        }
        return payload

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
<|endoftext|>from ._common import BaseConversationalTask


class PublicAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="publicai", base_url="https://api.publicai.co")
<|endoftext|>from ._common import BaseConversationalTask


class BasetenConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="baseten", base_url="https://inference.baseten.co")
<|endoftext|>from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping

from ._common import BaseConversationalTask


class FireworksAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="fireworks-ai", base_url="https://api.fireworks.ai")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/inference/v1/chat/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)
        response_format = parameters.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_details = response_format.get("json_schema")
            if isinstance(json_schema_details, dict) and "schema" in json_schema_details:
                payload["response_format"] = {  # type: ignore
                    "type": "json_object",
                    "schema": json_schema_details["schema"],
                }
        return payload
<|endoftext|>from typing import Any

from huggingface_hub.inference._common import RequestParameters, _as_dict

from ._common import BaseConversationalTask, InferenceProviderMapping, TaskProviderHelper, filter_none


class ScalewayConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="scaleway", base_url="https://api.scaleway.ai")


class ScalewayFeatureExtractionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="scaleway", base_url="https://api.scaleway.ai", task="feature-extraction")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/embeddings"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        parameters = filter_none(parameters)
        return {"input": inputs, "model": provider_mapping_info.provider_id, **parameters}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        embeddings = _as_dict(response)["data"]
        return [embedding["embedding"] for embedding in embeddings]
<|endoftext|>import base64
import time
from abc import ABC
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import (
    RequestParameters,
    _as_dict,
    _as_url,
)
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
)
from huggingface_hub.utils import get_session, hf_raise_for_status, logging


logger = logging.get_logger(__name__)

_PROVIDER = "together"
_BASE_URL = "https://api.together.xyz"

# Polling interval for async video generation (in seconds).
_VIDEO_POLLING_INTERVAL = 2.0

# Upper bound on status polls (initial response may already be terminal; each further poll is one attempt).
_VIDEO_MAX_POLL_ATTEMPTS = 150  # ~5 minutes at _VIDEO_POLLING_INTERVAL

# Job statuses that mean "keep polling". Together returns "queued" before transitioning to
# "in_progress", so we must treat both as pending.
_VIDEO_PENDING_STATUSES = {"queued", "in_progress"}


class TogetherTask(TaskProviderHelper, ABC):
    """Base class for Together API tasks."""

    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        match self.task:
            case "text-to-image" | "image-to-image":
                return "/v1/images/generations"
            case "text-to-speech":
                return "/v1/audio/speech"
            case "feature-extraction":
                return "/v1/embeddings"
            case "text-to-video" | "image-to-video":
                # Video creation lives under /v2 (see https://docs.together.ai/reference/create-videos).
                return "/v2/videos"
        raise ValueError(f"Unsupported task '{self.task}' for Together API.")


class TogetherTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class TogetherConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)
        if payload is None:
            return None
        # Together accepts response_format `{type: "json_schema", schema: <schema>}` (flattened),
        # so unwrap the OpenAI-style `{type: "json_schema", json_schema: {schema}}` envelope.
        response_format = payload.get("response_format")
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and isinstance(response_format.get("json_schema"), dict)
            and "schema" in response_format["json_schema"]
        ):
            payload["response_format"] = {
                "type": "json_schema",
                "schema": response_format["json_schema"]["schema"],
            }
        return payload


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")

        return {"prompt": inputs, "response_format": "base64", **parameters, "model": mapped_model}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])


class TogetherImageToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("image-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        image_url = _as_url(inputs, default_mime_type="image/jpeg")

        # Filter `None` values first: the client always passes `"prompt": None` when the user
        # omits the argument, so popping before filtering would yield `None` instead of the
        # `""` default and send `"prompt": null` to Together (rejected by Flux Kontext).
        parameters = filter_none(parameters)
        prompt = parameters.pop("prompt", "")
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")

        # Together exposes two mutually-exclusive image inputs (see
        # https://docs.together.ai/docs/image-to-image): FLUX.1 Kontext only accepts
        # `image_url`; FLUX.2 [dev] and Google models (Gemini 3 Pro Image, Flash Image
        # 2.5) only accept `reference_images`. FLUX.2 [pro]/[flex] accept either but
        # `reference_images` is the documented default. Use `image_url` only for
        # FLUX.1 Kontext models and `reference_images` for everything else.
        lowered = mapped_model.lower()
        use_image_url = "kontext" in lowered and "flux.1" in lowered
        image_field: dict[str, Any] = {"image_url": image_url} if use_image_url else {"reference_images": [image_url]}
        return {
            "prompt": prompt,
            **image_field,
            "response_format": "base64",
            **parameters,
            "model": mapped_model,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])


class TogetherFeatureExtractionTask(TogetherTask):
    def __init__(self):
        super().__init__("feature-extraction")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {
            "input": inputs,
            "model": provider_mapping_info.provider_id,
            **filter_none(parameters),
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        return [item["embedding"] for item in _as_dict(response)["data"]]


class TogetherTextToSpeechTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `voice` is required by the Together API and is model-specific
        # (see https://docs.together.ai/docs/text-to-speech#supported-voices),
        # so we don't set a default and let the API surface a clear error if missing.
        return {
            "input": inputs,
            "model": provider_mapping_info.provider_id,
            **filter_none(parameters),
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            return response
        raise ValueError(f"Expected raw audio bytes for text-to-speech, got {type(response).__name__}.")


def _normalize_video_parameters(parameters: dict) -> dict:
    """Map HF inference-client conventions onto Together's video API parameter names."""
    parameters = filter_none(parameters)
    if "num_inference_steps" in parameters:
        parameters["steps"] = parameters.pop("num_inference_steps")
    if "target_size" in parameters:
        target_size = parameters.pop("target_size")
        if "width" in target_size:
            parameters["width"] = target_size["width"]
        if "height" in target_size:
            parameters["height"] = target_size["height"]
    return parameters


class TogetherVideoTask(TogetherTask, ABC):
    """Base class for Together's asynchronous video generation tasks."""

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if request_params is None:
            raise ValueError("A `RequestParameters` object is required to poll Together video jobs.")

        job = _as_dict(response)
        job_id = job.get("id")
        if not job_id:
            raise ValueError("No job ID found in Together video generation response.")

        # Status polling lives at the same /v2/videos URL with the job ID appended.
        status_url = f"{request_params.url}/{job_id}"

        logger.info("Generating video, polling for completion...")
        # Together usually returns `status: "queued"` on the initial POST, but the field is
        # optional per the spec — treat a missing status as "still pending" and poll, rather
        # than falling through to the "unexpected status" error below.
        status = job.get("status")
        for _ in range(_VIDEO_MAX_POLL_ATTEMPTS):
            if status is not None and status not in _VIDEO_PENDING_STATUSES:
                break
            time.sleep(_VIDEO_POLLING_INTERVAL)
            status_response = get_session().get(status_url, headers=request_params.headers)
            hf_raise_for_status(status_response)
            job = status_response.json()
            status = job.get("status")
            if status is not None and status not in _VIDEO_PENDING_STATUSES:
                break
        else:
            raise ValueError(
                "Timed out while waiting for Together video generation "
                f"— aborting after {_VIDEO_MAX_POLL_ATTEMPTS} status polls"
            )

        if status == "failed":
            error = job.get("error") or {}
            raise RuntimeError(f"Together video generation failed: {error.get('message') or 'Unknown error'}")
        if status != "completed":
            raise RuntimeError(f"Unexpected Together video job status: {status!r}")

        video_url = (job.get("outputs") or {}).get("video_url")
        if not video_url:
            raise ValueError("No video URL found in completed Together video job.")

        video_response = get_session().get(video_url)
        hf_raise_for_status(video_response)
        return video_response.content


class TogetherTextToVideoTask(TogetherVideoTask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {
            "prompt": inputs,
            "model": provider_mapping_info.provider_id,
            **_normalize_video_parameters(parameters),
        }


class TogetherImageToVideoTask(TogetherVideoTask):
    def __init__(self):
        super().__init__("image-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # Together expects each keyframe as `{input_image, frame: "first" | "last"}`
        # for i2v models. See https://docs.together.ai/docs/inference/videos/reference-and-keyframes.
        # Note: `input_image` accepts a data URL or an HTTP(S) URL but the field is capped
        # at ~60KB — users with larger inputs should host the image and pass `frame_images`
        # directly via `extra_body`.
        return {
            "model": provider_mapping_info.provider_id,
            "frame_images": [{"input_image": _as_url(inputs, default_mime_type="image/png"), "frame": "first"}],
            **_normalize_video_parameters(parameters),
        }
<|endoftext|>from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping

from ._common import BaseConversationalTask


_PROVIDER = "cohere"
_BASE_URL = "https://api.cohere.com"


class CohereConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/compatibility/v1/chat/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)
        response_format = parameters.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_details = response_format.get("json_schema")
            if isinstance(json_schema_details, dict) and "schema" in json_schema_details:
                payload["response_format"] = {  # type: ignore
                    "type": "json_object",
                    "schema": json_schema_details["schema"],
                }

        return payload
<|endoftext|>import json
import mimetypes
import uuid
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import MimeBytes, RequestParameters, _as_dict, _open_as_mime_bytes

from ._common import BaseConversationalTask, BaseTextGenerationTask, TaskProviderHelper, filter_none


_PROVIDER = "deepinfra"
_BASE_URL = "https://api.deepinfra.com"


def _form_field_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _encode_multipart(audio: MimeBytes, fields: dict[str, Any]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    # Fall back to .wav when the MIME type is unknown: transcription servers sniff the format from the filename.
    filename = "audio" + (mimetypes.guess_extension(audio.mime_type or "") or ".wav")
    lines: list[bytes] = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {audio.mime_type or 'application/octet-stream'}".encode(),
        b"",
        bytes(audio),
    ]
    for key, value in fields.items():
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{key}"'.encode(),
            b"",
            _form_field_value(value).encode(),
        ]
    lines += [f"--{boundary}--".encode(), b""]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


class DeepInfraTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        params = filter_none(parameters.copy())
        params["max_tokens"] = params.pop("max_new_tokens", None)

        return {"prompt": inputs, **params, "model": provider_mapping_info.provider_id}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class DeepInfraConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/chat/completions"


class DeepInfraAutomaticSpeechRecognitionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="automatic-speech-recognition")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/audio/transcriptions"

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        # OpenAI-compatible transcription endpoint expects a multipart/form-data body, not JSON.
        audio = _open_as_mime_bytes(inputs)
        # `model` is applied last so parameters cannot override the mapped provider model.
        fields: dict[str, Any] = {
            **filter_none(parameters),
            **filter_none(extra_payload or {}),
            "model": provider_mapping_info.provider_id,
        }
        body, content_type = _encode_multipart(audio, fields)
        return MimeBytes(body, mime_type=content_type)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)
        text = output["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from DeepInfra API. Expected string, got {type(text)}.")
        result: dict[str, Any] = {"text": text}
        segments = output.get("segments")
        if isinstance(segments, list):
            result["chunks"] = [
                {"text": segment.get("text"), "timestamp": [segment.get("start"), segment.get("end")]}
                for segment in segments
                if isinstance(segment, dict)
            ]
        return result


class DeepInfraTextToSpeechTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-speech")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/audio/speech"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `voice` is model-specific and optional; we pass it through and let the API surface a
        # clear error when a model requires one. `model` is applied last so parameters cannot
        # override the mapped provider model.
        return {
            "input": inputs,
            **filter_none(parameters),
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            return response
        raise ValueError(f"Expected raw audio bytes for text-to-speech, got {type(response).__name__}.")


class DeepInfraFeatureExtractionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="feature-extraction")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/embeddings"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `model` is applied last so parameters cannot override the mapped provider model.
        return {
            "input": inputs,
            **filter_none(parameters),
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        return [item["embedding"] for item in _as_dict(response)["data"]]
<|endoftext|>from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._providers._common import BaseConversationalTask


class OpenAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="openai", base_url="https://api.openai.com")

    def _prepare_api_key(self, api_key: str | None) -> str:
        if api_key is None:
            raise ValueError("You must provide an api_key to work with OpenAI API.")
        if api_key.startswith("hf_"):
            raise ValueError(
                "OpenAI provider is not available through Hugging Face routing, please use your own OpenAI API key."
            )
        return api_key

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        if model is None:
            raise ValueError("Please provide an OpenAI model ID, e.g. `gpt-4o` or `o1`.")
        return InferenceProviderMapping(
            provider="openai", providerId=model, task="conversational", status="live", hf_model_id=model
        )
<|endoftext|>from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict

from ._common import BaseConversationalTask, BaseTextGenerationTask, filter_none


_PROVIDER = "featherless-ai"
_BASE_URL = "https://api.featherless.ai"


class FeatherlessTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        params = filter_none(parameters.copy())
        params["max_tokens"] = params.pop("max_new_tokens", None)

        return {"prompt": inputs, **params, "model": provider_mapping_info.provider_id}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class FeatherlessConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
<|endoftext|>from typing import Literal, Union

from huggingface_hub.inference._providers.featherless_ai import (
    FeatherlessConversationalTask,
    FeatherlessTextGenerationTask,
)
from huggingface_hub.utils import logging

from ._common import AutoRouterConversationalTask, TaskProviderHelper, _fetch_inference_provider_mapping
from .baseten import BasetenConversationalTask
from .cerebras import CerebrasConversationalTask
from .cohere import CohereConversationalTask
from .deepinfra import (
    DeepInfraAutomaticSpeechRecognitionTask,
    DeepInfraConversationalTask,
    DeepInfraFeatureExtractionTask,
    DeepInfraTextGenerationTask,
    DeepInfraTextToSpeechTask,
)
from .fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAIImageSegmentationTask,
    FalAIImageToImageTask,
    FalAIImageToVideoTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
    FalAITextToVideoTask,
)
from .fireworks_ai import FireworksAIConversationalTask
from .groq import GroqConversationalTask
from .hf_inference import (
    HFInferenceBinaryInputTask,
    HFInferenceConversational,
    HFInferenceFeatureExtractionTask,
    HFInferenceTask,
)
from .novita import NovitaConversationalTask, NovitaTextGenerationTask, NovitaTextToVideoTask
from .nscale import NscaleConversationalTask, NscaleTextToImageTask
from .openai import OpenAIConversationalTask
from .ovhcloud import OVHcloudConversationalTask
from .publicai import PublicAIConversationalTask
from .replicate import (
    ReplicateAutomaticSpeechRecognitionTask,
    ReplicateImageToImageTask,
    ReplicateTask,
    ReplicateTextToImageTask,
    ReplicateTextToSpeechTask,
)
from .scaleway import ScalewayConversationalTask, ScalewayFeatureExtractionTask
from .together import (
    TogetherConversationalTask,
    TogetherFeatureExtractionTask,
    TogetherImageToImageTask,
    TogetherImageToVideoTask,
    TogetherTextGenerationTask,
    TogetherTextToImageTask,
    TogetherTextToSpeechTask,
    TogetherTextToVideoTask,
)
from .wavespeed import (
    WavespeedAIImageToImageTask,
    WavespeedAIImageToVideoTask,
    WavespeedAITextToImageTask,
    WavespeedAITextToVideoTask,
)
from .zai_org import ZaiConversationalTask, ZaiTextToImageTask


logger = logging.get_logger(__name__)


PROVIDER_T = Literal[
    "baseten",
    "cerebras",
    "cohere",
    "deepinfra",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "novita",
    "nscale",
    "openai",
    "ovhcloud",
    "publicai",
    "replicate",
    "scaleway",
    "together",
    "wavespeed",
    "zai-org",
]

PROVIDER_OR_POLICY_T = Union[PROVIDER_T, Literal["auto"]]

CONVERSATIONAL_AUTO_ROUTER = AutoRouterConversationalTask()

PROVIDERS: dict[PROVIDER_T, dict[str, TaskProviderHelper]] = {
    "baseten": {
        "conversational": BasetenConversationalTask(),
    },
    "cerebras": {
        "conversational": CerebrasConversationalTask(),
    },
    "cohere": {
        "conversational": CohereConversationalTask(),
    },
    "deepinfra": {
        "automatic-speech-recognition": DeepInfraAutomaticSpeechRecognitionTask(),
        "conversational": DeepInfraConversationalTask(),
        "feature-extraction": DeepInfraFeatureExtractionTask(),
        "text-generation": DeepInfraTextGenerationTask(),
        "text-to-speech": DeepInfraTextToSpeechTask(),
    },
    "fal-ai": {
        "automatic-speech-recognition": FalAIAutomaticSpeechRecognitionTask(),
        "text-to-image": FalAITextToImageTask(),
        "text-to-speech": FalAITextToSpeechTask(),
        "text-to-video": FalAITextToVideoTask(),
        "image-to-video": FalAIImageToVideoTask(),
        "image-to-image": FalAIImageToImageTask(),
        "image-segmentation": FalAIImageSegmentationTask(),
    },
    "featherless-ai": {
        "conversational": FeatherlessConversationalTask(),
        "text-generation": FeatherlessTextGenerationTask(),
    },
    "fireworks-ai": {
        "conversational": FireworksAIConversationalTask(),
    },
    "groq": {
        "conversational": GroqConversationalTask(),
    },
    "hf-inference": {
        "text-to-image": HFInferenceTask("text-to-image"),
        "conversational": HFInferenceConversational(),
        "text-generation": HFInferenceTask("text-generation"),
        "text-classification": HFInferenceTask("text-classification"),
        "question-answering": HFInferenceTask("question-answering"),
        "audio-classification": HFInferenceBinaryInputTask("audio-classification"),
        "automatic-speech-recognition": HFInferenceBinaryInputTask("automatic-speech-recognition"),
        "fill-mask": HFInferenceTask("fill-mask"),
        "feature-extraction": HFInferenceFeatureExtractionTask(),
        "image-classification": HFInferenceBinaryInputTask("image-classification"),
        "image-segmentation": HFInferenceBinaryInputTask("image-segmentation"),
        "document-question-answering": HFInferenceTask("document-question-answering"),
        "image-to-text": HFInferenceBinaryInputTask("image-to-text"),
        "object-detection": HFInferenceBinaryInputTask("object-detection"),
        "audio-to-audio": HFInferenceBinaryInputTask("audio-to-audio"),
        "zero-shot-image-classification": HFInferenceBinaryInputTask("zero-shot-image-classification"),
        "zero-shot-classification": HFInferenceTask("zero-shot-classification"),
        "image-to-image": HFInferenceBinaryInputTask("image-to-image"),
        "sentence-similarity": HFInferenceTask("sentence-similarity"),
        "table-question-answering": HFInferenceTask("table-question-answering"),
        "tabular-classification": HFInferenceTask("tabular-classification"),
        "text-to-speech": HFInferenceTask("text-to-speech"),
        "token-classification": HFInferenceTask("token-classification"),
        "translation": HFInferenceTask("translation"),
        "summarization": HFInferenceTask("summarization"),
        "visual-question-answering": HFInferenceBinaryInputTask("visual-question-answering"),
    },
    "novita": {
        "text-generation": NovitaTextGenerationTask(),
        "conversational": NovitaConversationalTask(),
        "text-to-video": NovitaTextToVideoTask(),
    },
    "nscale": {
        "conversational": NscaleConversationalTask(),
        "text-to-image": NscaleTextToImageTask(),
    },
    "openai": {
        "conversational": OpenAIConversationalTask(),
    },
    "ovhcloud": {
        "conversational": OVHcloudConversationalTask(),
    },
    "publicai": {
        "conversational": PublicAIConversationalTask(),
    },
    "replicate": {
