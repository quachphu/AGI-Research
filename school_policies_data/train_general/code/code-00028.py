            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
        )

        # Case 5: generate constrained output using grammar
        >>> response = client.text_generation(
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
            bytes_output = self._inner_post(request_parameters, stream=stream or False)
        except HfHubHTTPError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(str(e))
            if isinstance(e, BadRequestError) and match:
                unused_params = [kwarg.strip("' ") for kwarg in match.group(1).split(",")]
                _set_unsupported_text_generation_kwargs(model, unused_params)
                return self.text_generation(  # type: ignore
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
            return _stream_text_generation_response(bytes_output, details)  # type: ignore

        data = _bytes_to_dict(bytes_output)  # type: ignore

        # Data can be a single element (dict) or an iterable of dicts where we select the first element of.
        if isinstance(data, list):
            data = data[0]
        response = provider_helper.get_response(data, request_parameters)
        return TextGenerationOutput.parse_obj_as_instance(response) if details else response["generated_text"]

    def text_to_image(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> image = client.text_to_image("An astronaut riding a horse on the moon.")
        >>> image.save("astronaut.png")

        >>> image = client.text_to_image(
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
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return _bytes_to_image(response)

    def text_to_video(
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
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return response

    def text_to_speech(
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
        >>> from pathlib import Path
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> audio = client.text_to_speech("Hello world")
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
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response)
        return response

    def token_classification(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
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
        response = self._inner_post(request_parameters)
        return TokenClassificationOutputElement.parse_obj_as_list(response)

    def translation(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.translation("My name is Wolfgang and I live in Berlin")
        'Mein Name ist Wolfgang und ich lebe in Berlin.'
        >>> client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
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
        response = self._inner_post(request_parameters)
        return TranslationOutput.parse_obj_as_list(response)[0]

    def visual_question_answering(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.visual_question_answering(
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
        response = self._inner_post(request_parameters)
        return VisualQuestionAnsweringOutputElement.parse_obj_as_list(response)

    def zero_shot_classification(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> text = (
        ...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
        ...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
        ...     " mysteries when he went for a run up a hill in Nice, France."
        ... )
        >>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
        >>> client.zero_shot_classification(text, labels)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
            ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
            ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
        ]
        >>> client.zero_shot_classification(text, labels, multi_label=True)
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.zero_shot_classification(
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
        response = self._inner_post(request_parameters)
        output = _bytes_to_dict(response)
        return ZeroShotClassificationOutputElement.parse_obj_as_list(output)

    def zero_shot_image_classification(
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> client.zero_shot_image_classification(
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
        response = self._inner_post(request_parameters)
        return ZeroShotImageClassificationOutputElement.parse_obj_as_list(response)

    def get_endpoint_info(self, *, model: str | None = None) -> dict[str, Any]:
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> client.get_endpoint_info()
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

        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        hf_raise_for_status(response)
        return response.json()

    def health_check(self, model: str | None = None) -> bool:
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
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
        >>> client.health_check()
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

        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        return response.status_code == 200

    @property
    def chat(self) -> "ProxyClientChat":
        return ProxyClientChat(self)


class _ProxyClient:
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    def __init__(self, client: InferenceClient):
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
"""Contains utilities used by both the sync and async inference clients."""

import base64
import io
import json
import logging
import mimetypes
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, NoReturn, Union, overload

import httpx2

from huggingface_hub.errors import (
    GenerationError,
    HfHubHTTPError,
    IncompleteGenerationError,
    OverloadedError,
    TextGenerationError,
    UnknownError,
    ValidationError,
)

from ..utils import get_session, is_numpy_available, is_pillow_available
from ._generated.types import ChatCompletionStreamOutput, TextGenerationStreamOutput


if TYPE_CHECKING:
    from PIL.Image import Image

# TYPES
UrlT = str
PathT = Union[str, Path]
ContentT = Union[bytes, BinaryIO, PathT, UrlT, "Image", bytearray, memoryview]

# Use to set an Accept: image/png header
TASKS_EXPECTING_IMAGES = {"text-to-image", "image-to-image"}

logger = logging.getLogger(__name__)


@dataclass
class RequestParameters:
    url: str
    task: str
    model: str | None
    json: str | dict | list | None
    data: bytes | None
    headers: dict[str, Any]


class MimeBytes(bytes):
    """
    A bytes object with a mime type.
    To be returned by `_prepare_payload_open_as_mime_bytes` in subclasses.

    Example:
    ```python
        >>> b = MimeBytes(b"hello", "text/plain")
        >>> isinstance(b, bytes)
        True
        >>> b.mime_type
        'text/plain'
    ```
    """

    mime_type: str | None

    def __new__(cls, data: bytes, mime_type: str | None = None):
        obj = super().__new__(cls, data)
        obj.mime_type = mime_type
        if isinstance(data, MimeBytes) and mime_type is None:
            obj.mime_type = data.mime_type
        return obj


## IMPORT UTILS


def _import_numpy():
    """Make sure `numpy` is installed on the machine."""
    if not is_numpy_available():
        raise ImportError("Please install numpy to use deal with embeddings (`pip install numpy`).")
    import numpy

    return numpy


def _import_pil_image():
    """Make sure `PIL` is installed on the machine."""
    if not is_pillow_available():
        raise ImportError(
            "Please install Pillow to use deal with images (`pip install Pillow`). If you don't want the image to be"
            " post-processed, use `client.post(...)` and get the raw response from the server."
        )
    from PIL import Image

    return Image


## ENCODING / DECODING UTILS


@overload
def _open_as_mime_bytes(content: ContentT) -> MimeBytes: ...  # means "if input is not None, output is not None"


@overload
def _open_as_mime_bytes(content: Literal[None]) -> Literal[None]: ...  # means "if input is None, output is None"


def _open_as_mime_bytes(content: ContentT | None) -> MimeBytes | None:
    """Open `content` as a binary file, either from a URL, a local path, raw bytes, or a PIL Image.

    Do nothing if `content` is None.
    """
    # If content is None, yield None
    if content is None:
        return None

    # If content is bytes, return it
    if isinstance(content, bytes):
        return MimeBytes(content)

    # If content is raw binary data (bytearray, memoryview)
    if isinstance(content, (bytearray, memoryview)):
        return MimeBytes(bytes(content))

    # If content is a binary file-like object
    if hasattr(content, "read"):  # duck-typing instead of isinstance(content, BinaryIO)
        logger.debug("Reading content from BinaryIO")
        data = content.read()
        mime_type = mimetypes.guess_type(str(content.name))[0] if hasattr(content, "name") else None
        if isinstance(data, str):
            raise TypeError("Expected binary stream (bytes), but got text stream")
        return MimeBytes(data, mime_type=mime_type)

    # If content is a string => must be either a URL or a path
    if isinstance(content, str):
        if content.startswith("https://") or content.startswith("http://"):
            logger.debug(f"Downloading content from {content}")
            response = get_session().get(content)
            mime_type = response.headers.get("Content-Type")
            if mime_type is None:
                mime_type = mimetypes.guess_type(content)[0]
            return MimeBytes(response.content, mime_type=mime_type)

        content = Path(content)
        if not content.exists():
            raise FileNotFoundError(
                f"File not found at {content}. If `data` is a string, it must either be a URL or a path to a local"
                " file. To pass raw content, please encode it as bytes first."
            )

    # If content is a Path => open it
    if isinstance(content, Path):
        logger.debug(f"Opening content from {content}")
        return MimeBytes(content.read_bytes(), mime_type=mimetypes.guess_type(content)[0])

    # If content is a PIL Image => convert to bytes
    if is_pillow_available():
        from PIL import Image

        if isinstance(content, Image.Image):
            logger.debug("Converting PIL Image to bytes")
            buffer = io.BytesIO()
            format = content.format or "PNG"
            content.save(buffer, format=format)
            return MimeBytes(buffer.getvalue(), mime_type=f"image/{format.lower()}")

    # If nothing matched, raise error
    raise TypeError(
        f"Unsupported content type: {type(content)}. "
        "Expected one of: bytes, bytearray, BinaryIO, memoryview, Path, str (URL or file path), or PIL.Image.Image."
    )


def _b64_encode(content: ContentT) -> str:
    """Encode a raw file (image, audio) into base64. Can be bytes, an opened file, a path or a URL."""
    raw_bytes = _open_as_mime_bytes(content)
    return base64.b64encode(raw_bytes).decode()


def _as_url(content: ContentT, default_mime_type: str) -> str:
    if isinstance(content, str) and content.startswith(("http://", "https://", "data:")):
        return content

    # Convert content to bytes
    raw_bytes = _open_as_mime_bytes(content)

    # Get MIME type
    mime_type = raw_bytes.mime_type or default_mime_type

    # Encode content to base64
    encoded_data = base64.b64encode(raw_bytes).decode()

    # Build data URL
    return f"data:{mime_type};base64,{encoded_data}"


def _b64_to_image(encoded_image: str) -> "Image":
    """Parse a base64-encoded string into a PIL Image."""
    Image = _import_pil_image()
    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def _bytes_to_list(content: bytes) -> list:
    """Parse bytes from a Response object into a Python list.

    Expects the response body to be JSON-encoded data.

    NOTE: This is exactly the same implementation as `_bytes_to_dict` and will not complain if the returned data is a
    dictionary. The only advantage of having both is to help the user (and mypy) understand what kind of data to expect.
    """
    return json.loads(content.decode())


def _bytes_to_dict(content: bytes) -> dict:
    """Parse bytes from a Response object into a Python dictionary.

    Expects the response body to be JSON-encoded data.

    NOTE: This is exactly the same implementation as `_bytes_to_list` and will not complain if the returned data is a
    list. The only advantage of having both is to help the user (and mypy) understand what kind of data to expect.
    """
    return json.loads(content.decode())


def _bytes_to_image(content: bytes) -> "Image":
    """Parse bytes from a Response object into a PIL Image.

    Expects the response body to be raw bytes. To deal with b64 encoded images, use `_b64_to_image` instead.
    """
    Image = _import_pil_image()
    return Image.open(io.BytesIO(content))


def _as_dict(response: bytes | dict) -> dict:
    return json.loads(response) if isinstance(response, bytes) else response


## STREAMING UTILS


def _stream_text_generation_response(
    output_lines: Iterable[str], details: bool
) -> Iterable[str] | Iterable[TextGenerationStreamOutput]:
    """Used in `InferenceClient.text_generation`."""
    # Parse ServerSentEvents
    for line in output_lines:
        try:
            output = _format_text_generation_stream_output(line, details)
        except StopIteration:
            break
        if output is not None:
            yield output


async def _async_stream_text_generation_response(
    output_lines: AsyncIterable[str], details: bool
) -> AsyncIterable[str] | AsyncIterable[TextGenerationStreamOutput]:
    """Used in `AsyncInferenceClient.text_generation`."""
    # Parse ServerSentEvents
    async for line in output_lines:
        try:
            output = _format_text_generation_stream_output(line, details)
        except StopIteration:
            break
        if output is not None:
            yield output


def _format_text_generation_stream_output(line: str, details: bool) -> str | TextGenerationStreamOutput | None:
    if not line.startswith("data:"):
        return None  # empty line

    if line.strip() == "data: [DONE]":
        raise StopIteration("[DONE] signal received.")

    # Decode payload
    payload = line.lstrip("data:").rstrip("/n")
    json_payload = json.loads(payload)

    # Either an error as being returned
    if json_payload.get("error") is not None:
        raise _parse_text_generation_error(json_payload["error"], json_payload.get("error_type"))

    # Or parse token payload
    output = TextGenerationStreamOutput.parse_obj_as_instance(json_payload)
    return output.token.text if not details else output


def _stream_chat_completion_response(
    lines: Iterable[str],
) -> Iterable[ChatCompletionStreamOutput]:
    """Used in `InferenceClient.chat_completion` if model is served with TGI."""
    for line in lines:
        try:
            output = _format_chat_completion_stream_output(line)
        except StopIteration:
            break
        if output is not None:
            yield output


async def _async_stream_chat_completion_response(
    lines: AsyncIterable[str],
) -> AsyncIterable[ChatCompletionStreamOutput]:
    """Used in `AsyncInferenceClient.chat_completion`."""
    async for line in lines:
        try:
            output = _format_chat_completion_stream_output(line)
        except StopIteration:
            break
        if output is not None:
            yield output


def _format_chat_completion_stream_output(
    line: str,
) -> ChatCompletionStreamOutput | None:
    if not line.startswith("data:"):
        return None  # empty line

    if line.strip() == "data: [DONE]":
        raise StopIteration("[DONE] signal received.")

    # Decode payload
    json_payload = json.loads(line.lstrip("data:").strip())

    # Either an error as being returned
    if json_payload.get("error") is not None:
        raise _parse_text_generation_error(json_payload["error"], json_payload.get("error_type"))

    # Or parse token payload
    return ChatCompletionStreamOutput.parse_obj_as_instance(json_payload)


async def _async_yield_from(client: httpx2.AsyncClient, response: httpx2.Response) -> AsyncIterable[str]:
    async for line in response.aiter_lines():
        yield line.strip()


# "TGI servers" are servers running with the `text-generation-inference` backend.
# This backend is the go-to solution to run large language models at scale. However,
# for some smaller models (e.g. "gpt2") the default `transformers` + `api-inference`
# solution is still in use.
#
# Both approaches have very similar APIs, but not exactly the same. What we do first in
# the `text_generation` method is to assume the model is served via TGI. If we realize
# it's not the case (i.e. we receive an HTTP 400 Bad Request), we fall back to the
# default API with a warning message. When that's the case, We remember the unsupported
# attributes for this model in the `_UNSUPPORTED_TEXT_GENERATION_KWARGS` global variable.
#
# In addition, TGI servers have a built-in API route for chat-completion, which is not
# available on the default API. We use this route to provide a more consistent behavior
# when available.
#
# For more details, see https://github.com/huggingface/text-generation-inference and
# https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task.

_UNSUPPORTED_TEXT_GENERATION_KWARGS: dict[str | None, list[str]] = {}


def _set_unsupported_text_generation_kwargs(model: str | None, unsupported_kwargs: list[str]) -> None:
    _UNSUPPORTED_TEXT_GENERATION_KWARGS.setdefault(model, []).extend(unsupported_kwargs)


def _get_unsupported_text_generation_kwargs(model: str | None) -> list[str]:
    return _UNSUPPORTED_TEXT_GENERATION_KWARGS.get(model, [])


# TEXT GENERATION ERRORS
# ----------------------
# Text-generation errors are parsed separately to handle as much as possible the errors returned by the text generation
# inference project (https://github.com/huggingface/text-generation-inference).
# ----------------------


def raise_text_generation_error(http_error: HfHubHTTPError) -> NoReturn:
    """
    Try to parse text-generation-inference error message and raise HTTPError in any case.

    Args:
        http_error (`HfHubHTTPError`):
            The HTTPError that has been raised.
    """
    # Try to parse a Text Generation Inference error
    if http_error.response is None:
        raise http_error

    try:
        # Hacky way to retrieve payload in case of aiohttp error
        payload = getattr(http_error, "response_error_payload", None) or http_error.response.json()
        error = payload.get("error")
        error_type = payload.get("error_type")
    except Exception:  # no payload
        raise http_error

    # If error_type => more information than `hf_raise_for_status`
    if error_type is not None:
        exception = _parse_text_generation_error(error, error_type)
        raise exception from http_error

    # Otherwise, fallback to default error
    raise http_error


def _parse_text_generation_error(error: str | None, error_type: str | None) -> TextGenerationError:
    if error_type == "generation":
        return GenerationError(error)  # type: ignore
    if error_type == "incomplete_generation":
        return IncompleteGenerationError(error)  # type: ignore
    if error_type == "overloaded":
        return OverloadedError(error)  # type: ignore
    if error_type == "validation":
        return ValidationError(error)  # type: ignore
    return UnknownError(error)  # type: ignore
<|endoftext|>import asyncio
import sys
from functools import partial

import click


def _patch_anyio_open_process():
    """
    Patch anyio.open_process to allow detached processes on Windows and Unix-like systems.

    This is necessary to prevent the MCP client from being interrupted by Ctrl+C when running in the CLI.
    """
    import subprocess

    import anyio

    if getattr(anyio, "_tiny_agents_patched", False):
        return
    anyio._tiny_agents_patched = True  # ty: ignore[invalid-assignment]

    original_open_process = anyio.open_process

    if sys.platform == "win32":
        # On Windows, we need to set the creation flags to create a new process group

        async def open_process_in_new_group(*args, **kwargs):
            """
            Wrapper for open_process to handle Windows-specific process creation flags.
            """
            # Ensure we pass the creation flags for Windows
            kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
            return await original_open_process(*args, **kwargs)

        anyio.open_process = open_process_in_new_group  # ty: ignore[invalid-assignment]
    else:
        # For Unix-like systems, we can use setsid to create a new session
        async def open_process_in_new_group(*args, **kwargs):
            """
            Wrapper for open_process to handle Unix-like systems with start_new_session=True.
            """
            kwargs.setdefault("start_new_session", True)
            return await original_open_process(*args, **kwargs)

        anyio.open_process = open_process_in_new_group  # ty: ignore[invalid-assignment]


async def _async_prompt(exit_event: asyncio.Event, prompt: str = "» ") -> str:
    """
    Asynchronous prompt function that reads input from stdin without blocking.

    This function is designed to work in an asynchronous context, allowing the event loop to gracefully stop it (e.g. on Ctrl+C).

    Alternatively, we could use https://github.com/vxgmichel/aioconsole but that would be an additional dependency.
    """
    loop = asyncio.get_event_loop()

    if sys.platform == "win32":
        # Windows: Use run_in_executor to avoid blocking the event loop
        # Degraded solution: this is not ideal as user will have to CTRL+C once more to stop the prompt (and it'll not be graceful)
        return await loop.run_in_executor(None, partial(click.prompt, prompt, prompt_suffix=" "))
    else:
        # UNIX-like: Use loop.add_reader for non-blocking stdin read
        future = loop.create_future()

        def on_input():
            line = sys.stdin.readline()
            loop.remove_reader(sys.stdin)
            future.set_result(line)

        print(prompt, end=" ", flush=True)
        loop.add_reader(sys.stdin, on_input)  # not supported on Windows

        # Wait for user input or exit event
        # Wait until either the user hits enter or exit_event is set
        exit_task = asyncio.create_task(exit_event.wait())
        await asyncio.wait(
            [future, exit_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Check which one has been triggered
        if exit_event.is_set():
            future.cancel()
            return ""

        line = await future
        return line.strip()
<|endoftext|>from typing import Literal, TypedDict, Union

from typing_extensions import NotRequired


class InputConfig(TypedDict, total=False):
    id: str
    description: str
    type: str
    password: bool


class StdioServerConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str
    allowed_tools: NotRequired[list[str]]


class HTTPServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: dict[str, str]
    allowed_tools: NotRequired[list[str]]


class SSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: dict[str, str]
    allowed_tools: NotRequired[list[str]]


ServerConfig = Union[StdioServerConfig, HTTPServerConfig, SSEServerConfig]


# AgentConfig root object
class AgentConfig(TypedDict):
    model: str
    provider: str
    apiKey: NotRequired[str]
    inputs: list[InputConfig]
    servers: list[ServerConfig]
<|endoftext|>"""
Utility functions for MCPClient and Tiny Agents.

Formatting utilities taken from the JS SDK: https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/ResultFormatter.ts.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from .constants import DEFAULT_AGENT, DEFAULT_REPO_ID, FILENAME_CONFIG, PROMPT_FILENAMES
from .types import AgentConfig


if TYPE_CHECKING:
    from mcp import types as mcp_types


def format_result(result: "mcp_types.CallToolResult") -> str:
    """
    Formats a mcp.types.CallToolResult content into a human-readable string.

    Args:
        result (CallToolResult)
            Object returned by mcp.ClientSession.call_tool.

    Returns:
        str
            A formatted string representing the content of the result.
    """
    content = result.content

    if len(content) == 0:
        return "[No content]"

    formatted_parts: list[str] = []

    for item in content:
        match item.type:
            case "text":
                formatted_parts.append(item.text)

            case "image":
                formatted_parts.append(
                    f"[Binary Content: Image {item.mimeType}, {_get_base64_size(item.data)} bytes]\n"
                    f"The task is complete and the content accessible to the User"
                )

            case "audio":
                formatted_parts.append(
                    f"[Binary Content: Audio {item.mimeType}, {_get_base64_size(item.data)} bytes]\n"
                    f"The task is complete and the content accessible to the User"
                )

            case "resource":
                resource = item.resource

                if hasattr(resource, "text") and isinstance(resource.text, str):
                    formatted_parts.append(resource.text)

                elif hasattr(resource, "blob") and isinstance(resource.blob, str):
                    formatted_parts.append(
                        f"[Binary Content ({resource.uri}): {resource.mimeType},"
                        f" {_get_base64_size(resource.blob)} bytes]\n"
                        f"The task is complete and the content accessible to the User"
                    )

    return "\n".join(formatted_parts)


def _get_base64_size(base64_str: str) -> int:
    """Estimate the byte size of a base64-encoded string."""
    # Remove any prefix like "data:image/png;base64,"
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    padding = 0
    if base64_str.endswith("=="):
        padding = 2
    elif base64_str.endswith("="):
        padding = 1

    return (len(base64_str) * 3) // 4 - padding


def _load_agent_config(agent_path: Optional[str]) -> tuple[AgentConfig, Optional[str]]:
    """Load server config and prompt."""

    def _read_dir(directory: Path) -> tuple[AgentConfig, Optional[str]]:
        cfg_file = directory / FILENAME_CONFIG
        if not cfg_file.exists():
            raise FileNotFoundError(f" Config file not found in {directory}! Please make sure it exists locally")

        config: AgentConfig = json.loads(cfg_file.read_text(encoding="utf-8"))
        prompt: Optional[str] = None
        for filename in PROMPT_FILENAMES:
            prompt_file = directory / filename
            if prompt_file.exists():
                prompt = prompt_file.read_text(encoding="utf-8")
                break
        return config, prompt

    if agent_path is None:
        return DEFAULT_AGENT, None  # type: ignore

    path = Path(agent_path).expanduser()

    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), None

    if path.is_dir():
        return _read_dir(path)

    # fetch from the Hub
    try:
        repo_dir = Path(
            snapshot_download(
                repo_id=DEFAULT_REPO_ID,
                allow_patterns=f"{agent_path}/*",
                repo_type="dataset",
            )
        )
        return _read_dir(repo_dir / agent_path)
    except Exception as err:
        raise EntryNotFoundError(
            f" Agent {agent_path} not found in tiny-agents/tiny-agents! Please make sure it exists in https://huggingface.co/datasets/tiny-agents/tiny-agents."
        ) from err
<|endoftext|>from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import ChatCompletionInputTool


FILENAME_CONFIG = "agent.json"
PROMPT_FILENAMES = ("PROMPT.md", "AGENTS.md")

DEFAULT_AGENT = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "novita",
    "servers": [
        {
            "type": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
            ],
        },
        {
            "type": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest"],
        },
    ],
}


DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user’s query is completely
resolved, before ending your turn and yielding back to the user. Only terminate
your turn when you are sure that the problem is solved, or if you need more
info from the user to solve the problem.
If you are not sure about anything pertaining to the user’s request, use your
tools to read files and gather the relevant information: do NOT guess or make
up an answer.
You MUST plan extensively before each function call, and reflect extensively
on the outcomes of the previous function calls. DO NOT do this entire process
by making function calls only, as this can impair your ability to solve the
problem and think insightfully.
""".strip()

MAX_NUM_TURNS = 10

TASK_COMPLETE_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(  # type: ignore
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this tool when the task given by the user is complete",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
)

ASK_QUESTION_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(  # type: ignore
    {
        "type": "function",
        "function": {
            "name": "ask_question",
            "description": "Ask the user for more info required to solve or clarify their problem.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
)

EXIT_LOOP_TOOLS: list[ChatCompletionInputTool] = [TASK_COMPLETE_TOOL, ASK_QUESTION_TOOL]


DEFAULT_REPO_ID = "tiny-agents/tiny-agents"
<|endoftext|>import json
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterable, Literal, Optional, TypedDict, Union, overload

from typing_extensions import NotRequired, TypeAlias, Unpack

from ...utils._runtime import get_hf_hub_version
from .._generated._async_client import AsyncInferenceClient
from .._generated.types import (
    ChatCompletionInputMessage,
    ChatCompletionInputTool,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputDeltaToolCall,
)
from .._providers import PROVIDER_OR_POLICY_T
from .utils import format_result


if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)

# Type alias for tool names
ToolName: TypeAlias = str

ServerType: TypeAlias = Literal["stdio", "sse", "http"]


class StdioServerParameters_T(TypedDict):
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
    cwd: NotRequired[Union[str, Path, None]]


class SSEServerParameters_T(TypedDict):
    url: str
    headers: NotRequired[dict[str, Any]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


class StreamableHTTPParameters_T(TypedDict):
    url: str
    headers: NotRequired[dict[str, Any]]
    timeout: NotRequired[timedelta]
    sse_read_timeout: NotRequired[timedelta]
    terminate_on_close: NotRequired[bool]


class MCPClient:
    """
    Client for connecting to one or more MCP servers and processing chat completions with tools.

    > [!WARNING]
    > This class is experimental and might be subject to breaking changes in the future without prior notice.

    Args:
        model (`str`, `optional`):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `meta-llama/Meta-Llama-3-8B-Instruct`
            or a URL to a deployed Inference Endpoint or other local or remote endpoint.
        provider (`str`, *optional*):
            Name of the provider to use for inference. Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
            If model is a URL or `base_url` is passed, then `provider` is not used.
        base_url (`str`, *optional*):
            The base URL to run inference. Defaults to None.
        api_key (`str`, `optional`):
            Token to use for authentication. Will default to the locally Hugging Face saved token if not provided. You can also use your own provider API key to interact directly with the provider's service.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Initialize MCP sessions as a dictionary of ClientSession objects
        self.sessions: dict[ToolName, "ClientSession"] = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools: list[ChatCompletionInputTool] = []
        # To be able to send the model in the payload if `base_url` is provided
        if model is None and base_url is None:
            raise ValueError("At least one of `model` or `base_url` should be set in `MCPClient`.")
        self.payload_model = model
        self.client = AsyncInferenceClient(
            model=None if base_url is not None else model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

    async def __aenter__(self):
        """Enter the context manager"""
        await self.client.__aenter__()
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager"""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        await self.client.close()
        await self.exit_stack.aclose()

    @overload
    async def add_mcp_server(self, type: Literal["stdio"], **params: Unpack[StdioServerParameters_T]): ...

    @overload
    async def add_mcp_server(self, type: Literal["sse"], **params: Unpack[SSEServerParameters_T]): ...

    @overload
    async def add_mcp_server(self, type: Literal["http"], **params: Unpack[StreamableHTTPParameters_T]): ...

    async def add_mcp_server(self, type: ServerType, **params: Any):
        """Connect to an MCP server

        Args:
            type (`str`):
                Type of the server to connect to. Can be one of:
                - "stdio": Standard input/output server (local)
                - "sse": Server-sent events (SSE) server
                - "http": StreamableHTTP server
            **params (`dict[str, Any]`):
                Server parameters that can be either:
                    - For stdio servers:
                        - command (str): The command to run the MCP server
                        - args (list[str], optional): Arguments for the command
                        - env (dict[str, str], optional): Environment variables for the command
                        - cwd (Union[str, Path, None], optional): Working directory for the command
                        - allowed_tools (list[str], optional): List of tool names to allow from this server
                    - For SSE servers:
                        - url (str): The URL of the SSE server
                        - headers (dict[str, Any], optional): Headers for the SSE connection
                        - timeout (float, optional): Connection timeout
                        - sse_read_timeout (float, optional): SSE read timeout
                        - allowed_tools (list[str], optional): List of tool names to allow from this server
                    - For StreamableHTTP servers:
                        - url (str): The URL of the StreamableHTTP server
                        - headers (dict[str, Any], optional): Headers for the StreamableHTTP connection
                        - timeout (timedelta, optional): Connection timeout
                        - sse_read_timeout (timedelta, optional): SSE read timeout
                        - terminate_on_close (bool, optional): Whether to terminate on close
                        - allowed_tools (list[str], optional): List of tool names to allow from this server
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp import types as mcp_types

        # Extract allowed_tools configuration if provided
        allowed_tools = params.pop("allowed_tools", None)

        # Determine server type and create appropriate parameters
        if type == "stdio":
            # Handle stdio server
            from mcp.client.stdio import stdio_client

            logger.info(f"Connecting to stdio MCP server with command: {params['command']} {params.get('args', [])}")

            client_kwargs = {"command": params["command"]}
            for key in ["args", "env", "cwd"]:
                if params.get(key) is not None:
                    client_kwargs[key] = params[key]
            server_params = StdioServerParameters(**client_kwargs)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        elif type == "sse":
            # Handle SSE server
            from mcp.client.sse import sse_client

            logger.info(f"Connecting to SSE MCP server at: {params['url']}")

            client_kwargs = {"url": params["url"]}
            for key in ["headers", "timeout", "sse_read_timeout"]:
                if params.get(key) is not None:
                    client_kwargs[key] = params[key]
            read, write = await self.exit_stack.enter_async_context(sse_client(**client_kwargs))
        elif type == "http":
            # Handle StreamableHTTP server
            from mcp.client.streamable_http import streamablehttp_client

            logger.info(f"Connecting to StreamableHTTP MCP server at: {params['url']}")

            client_kwargs = {"url": params["url"]}
            for key in ["headers", "timeout", "sse_read_timeout", "terminate_on_close"]:
                if params.get(key) is not None:
                    client_kwargs[key] = params[key]
            read, write, _ = await self.exit_stack.enter_async_context(streamablehttp_client(**client_kwargs))
            # ^ TODO: should be handle `get_session_id_callback`? (function to retrieve the current session ID)
        else:
            raise ValueError(f"Unsupported server type: {type}")

        session = await self.exit_stack.enter_async_context(
            ClientSession(
                read_stream=read,
                write_stream=write,
                client_info=mcp_types.Implementation(
                    name="huggingface_hub.MCPClient",
                    version=get_hf_hub_version(),
                ),
            )
        )

        logger.debug("Initializing session...")
        await session.initialize()

        # List available tools, following `nextCursor` since servers may paginate tools/list
        response = await session.list_tools()
        tools: list[mcp_types.Tool] = list(response.tools)
        while response.nextCursor:
            response = await session.list_tools(cursor=response.nextCursor)
            tools.extend(response.tools)
        logger.debug("Connected to server with tools:", [tool.name for tool in tools])

        # Filter tools based on allowed_tools configuration
        filtered_tools = tools

        if allowed_tools is not None:
            filtered_tools = [tool for tool in tools if tool.name in allowed_tools]
            logger.debug(
                f"Tool filtering applied. Using {len(filtered_tools)} of {len(tools)} available tools: {[tool.name for tool in filtered_tools]}"
            )

        for tool in filtered_tools:
            if tool.name in self.sessions:
                logger.warning(f"Tool '{tool.name}' already defined by another server. Skipping.")
                continue

            # Map tool names to their server for later lookup
            self.sessions[tool.name] = session

            # Add tool to the list of available tools (for use in chat completions)
            self.available_tools.append(
                ChatCompletionInputTool.parse_obj_as_instance(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
            )

    async def process_single_turn_with_tools(
        self,
        messages: list[Union[dict, ChatCompletionInputMessage]],
        exit_loop_tools: Optional[list[ChatCompletionInputTool]] = None,
        exit_if_first_chunk_no_tool: bool = False,
    ) -> AsyncIterable[Union[ChatCompletionStreamOutput, ChatCompletionInputMessage]]:
        """Process a query using `self.model` and available tools, yielding chunks and tool outputs.

        Args:
            messages (`list[dict]`):
                List of message objects representing the conversation history
            exit_loop_tools (`list[ChatCompletionInputTool]`, *optional*):
                List of tools that should exit the generator when called
            exit_if_first_chunk_no_tool (`bool`, *optional*):
                Exit if no tool is present in the first chunks. Default to False.

        Yields:
            [`ChatCompletionStreamOutput`] chunks or [`ChatCompletionInputMessage`] objects
        """
        # Prepare tools list based on options
        tools = self.available_tools
        if exit_loop_tools is not None:
            tools = [*exit_loop_tools, *self.available_tools]

        # Create the streaming request
        response = await self.client.chat.completions.create(
            model=self.payload_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        message: dict[str, Any] = {"role": "unknown", "content": ""}
        final_tool_calls: dict[int, ChatCompletionStreamOutputDeltaToolCall] = {}
        num_of_chunks = 0

        # Read from stream
        async for chunk in response:
            num_of_chunks += 1
            delta = chunk.choices[0].delta if chunk.choices and len(chunk.choices) > 0 else None
            if not delta:
                continue

            # Process message
            if delta.role:
                message["role"] = delta.role
            if delta.content:
                message["content"] += delta.content

            # Process tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    # first chunk for this tool call
                    if idx not in final_tool_calls:
                        final_tool_calls[idx] = tool_call
                        if final_tool_calls[idx].function.arguments is None:
                            final_tool_calls[idx].function.arguments = ""
                        continue
                    # safety before concatenating text to .function.arguments
                    if final_tool_calls[idx].function.arguments is None:
                        final_tool_calls[idx].function.arguments = ""

                    if tool_call.function.arguments:
                        final_tool_calls[idx].function.arguments += tool_call.function.arguments

            # Optionally exit early if no tools in first chunks
            if exit_if_first_chunk_no_tool and num_of_chunks <= 2 and len(final_tool_calls) == 0:
                return

            # Yield each chunk to caller
            yield chunk

        # Add the assistant message with tool calls (if any) to messages
        if message["content"] or final_tool_calls:
            # if the role is unknown, set it to assistant
            if message.get("role") == "unknown":
                message["role"] = "assistant"
            # Convert final_tool_calls to the format expected by OpenAI
            if final_tool_calls:
                tool_calls_list: list[dict[str, Any]] = []
                for tc in final_tool_calls.values():
                    tool_calls_list.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                    )
                message["tool_calls"] = tool_calls_list
            messages.append(message)

        # Process tool calls one by one
        for tool_call in final_tool_calls.values():
            function_name = tool_call.function.name
            if function_name is None:
                message = ChatCompletionInputMessage.parse_obj_as_instance(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Invalid tool call with no function name.",
                    }
                )
                messages.append(message)
                yield message
                continue  # move to next tool call
            try:
                function_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as err:
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": f"Invalid JSON generated by the model: {err}",
                }
                tool_message_as_obj = ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
                messages.append(tool_message_as_obj)
                yield tool_message_as_obj
                continue  # move to next tool call

            tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": "", "name": function_name}

            # Check if this is an exit loop tool
            if exit_loop_tools and function_name in [t.function.name for t in exit_loop_tools]:
                tool_message_as_obj = ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
                messages.append(tool_message_as_obj)
                yield tool_message_as_obj
                return

            # Execute tool call with the appropriate session
            session = self.sessions.get(function_name)
            if session is not None:
                try:
                    result = await session.call_tool(function_name, function_args)
                    tool_message["content"] = format_result(result)
                except Exception as err:
                    tool_message["content"] = f"Error: MCP tool call failed with error message: {err}"
            else:
                tool_message["content"] = f"Error: No session found for tool: {function_name}"

            # Yield tool message
            tool_message_as_obj = ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
            messages.append(tool_message_as_obj)
            yield tool_message_as_obj
<|endoftext|>from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Iterable, Optional, Union

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient

from .._providers import PROVIDER_OR_POLICY_T
from .constants import DEFAULT_SYSTEM_PROMPT, EXIT_LOOP_TOOLS, MAX_NUM_TURNS
from .types import ServerConfig


class Agent(MCPClient):
    """
    Implementation of a Simple Agent, which is a simple while loop built right on top of an [`MCPClient`].

    > [!WARNING]
    > This class is experimental and might be subject to breaking changes in the future without prior notice.

    Args:
        model (`str`, *optional*):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `meta-llama/Meta-Llama-3-8B-Instruct`
            or a URL to a deployed Inference Endpoint or other local or remote endpoint.
        servers (`Iterable[dict]`):
            MCP servers to connect to. Each server is a dictionary containing a `type` key and a `config` key. The `type` key can be `"stdio"` or `"sse"`, and the `config` key is a dictionary of arguments for the server.
        provider (`str`, *optional*):
            Name of the provider to use for inference. Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
            If model is a URL or `base_url` is passed, then `provider` is not used.
        base_url (`str`, *optional*):
            The base URL to run inference. Defaults to None.
        api_key (`str`, *optional*):
            Token to use for authentication. Will default to the locally Hugging Face saved token if not provided. You can also use your own provider API key to interact directly with the provider's service.
        prompt (`str`, *optional*):
            The system prompt to use for the agent. Defaults to the default system prompt in `constants.py`.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        servers: Iterable[ServerConfig],
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(model=model, provider=provider, base_url=base_url, api_key=api_key)
        self._servers_cfg = list(servers)
        self.messages: list[Union[dict, ChatCompletionInputMessage]] = [
            {"role": "system", "content": prompt or DEFAULT_SYSTEM_PROMPT}
        ]

    async def load_tools(self) -> None:
        for cfg in self._servers_cfg:
            await self.add_mcp_server(**cfg)

    async def run(
        self,
        user_input: str,
        *,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Union[ChatCompletionStreamOutput, ChatCompletionInputMessage], None]:
        """
        Run the agent with the given user input.

        Args:
            user_input (`str`):
                The user input to run the agent with.
            abort_event (`asyncio.Event`, *optional*):
                An event that can be used to abort the agent. If the event is set, the agent will stop running.
        """
        self.messages.append({"role": "user", "content": user_input})

        num_turns: int = 0
        next_turn_should_call_tools = True

        while True:
            if abort_event and abort_event.is_set():
                return

            async for item in self.process_single_turn_with_tools(
                self.messages,
                exit_loop_tools=EXIT_LOOP_TOOLS,
                exit_if_first_chunk_no_tool=(num_turns > 0 and next_turn_should_call_tools),
            ):
                yield item

            num_turns += 1
            last = self.messages[-1]

            if last.get("role") == "tool" and last.get("name") in {t.function.name for t in EXIT_LOOP_TOOLS}:
                return

            if last.get("role") != "tool" and num_turns > MAX_NUM_TURNS:
                return

            if last.get("role") != "tool" and next_turn_should_call_tools:
                return

            next_turn_should_call_tools = last.get("role") != "tool"
<|endoftext|>import asyncio
import os
import signal
import traceback
from typing import Optional

import click

from ...utils import ANSI
from ._cli_hacks import _async_prompt, _patch_anyio_open_process
from .agent import Agent
from .utils import _load_agent_config


@click.group(
    help="A squad of lightweight composable AI applications built on Hugging Face's Inference Client and MCP stack."
)
def app() -> None:
    pass


async def run_agent(
    agent_path: Optional[str],
) -> None:
    """
    Tiny Agent loop.

    Args:
        agent_path (`str`, *optional*):
            Path to a local folder containing an `agent.json` and optionally a custom `PROMPT.md` or `AGENTS.md` file or a built-in agent stored in a Hugging Face dataset.

    """
    _patch_anyio_open_process()  # Hacky way to prevent stdio connections to be stopped by Ctrl+C

    config, prompt = _load_agent_config(agent_path)

    inputs = config.get("inputs", [])
    servers = config.get("servers", [])

    abort_event = asyncio.Event()
    exit_event = asyncio.Event()
    first_sigint = True

    loop = asyncio.get_running_loop()
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler() -> None:
        nonlocal first_sigint
        if first_sigint:
            first_sigint = False
            abort_event.set()
            print(ANSI.red("\nInterrupted. Press Ctrl+C again to quit."), flush=True)
            return

        print(ANSI.red("\nExiting..."), flush=True)
        exit_event.set()

    try:
        sigint_registered_in_loop = False
        try:
            loop.add_signal_handler(signal.SIGINT, _sigint_handler)
            sigint_registered_in_loop = True
        except (AttributeError, NotImplementedError):
            # Windows (or any loop that doesn't support it) : fall back to sync
            signal.signal(signal.SIGINT, lambda *_: _sigint_handler())

        # Handle inputs (i.e. env variables injection)
        resolved_inputs: dict[str, str] = {}

        if len(inputs) > 0:
            print(
                ANSI.bold(
                    ANSI.blue(
                        "Some initial inputs are required by the agent. "
                        "Please provide a value or leave empty to load from env."
                    )
                )
            )
            for input_item in inputs:
                input_id = input_item["id"]
                description = input_item["description"]
                env_special_value = f"${{input:{input_id}}}"

                # Check if the input is used by any server or as an apiKey
                input_usages = set()
                for server in servers:
                    # Check stdio's "env" and http/sse's "headers" mappings
                    env_or_headers = server.get("env", {}) if server["type"] == "stdio" else server.get("headers", {})
                    for key, value in env_or_headers.items():
                        if env_special_value in value:
                            input_usages.add(key)

                raw_api_key = config.get("apiKey")
                if isinstance(raw_api_key, str) and env_special_value in raw_api_key:
                    input_usages.add("apiKey")

                if not input_usages:
                    print(
                        ANSI.yellow(
                            f"Input '{input_id}' defined in config but not used by any server or as an API key."
                            " Skipping."
                        )
                    )
                    continue

                # Prompt user for input
                env_variable_key = input_id.replace("-", "_").upper()
                print(
                    ANSI.blue(f" • {input_id}") + f": {description}. (default: load from {env_variable_key}).",
                    end=" ",
                )
                user_input = (await _async_prompt(exit_event=exit_event)).strip()
                if exit_event.is_set():
                    return

                # Fallback to environment variable when user left blank
                final_value = user_input
                if not final_value:
                    final_value = os.getenv(env_variable_key, "")
                    if final_value:
                        print(ANSI.green(f"Value successfully loaded from '{env_variable_key}'"))
                    else:
                        print(
                            ANSI.yellow(
                                f"No value found for '{env_variable_key}' in environment variables. Continuing."
                            )
                        )
                resolved_inputs[input_id] = final_value

                # Inject resolved value (can be empty) into stdio's env or http/sse's headers
                for server in servers:
                    env_or_headers = server.get("env", {}) if server["type"] == "stdio" else server.get("headers", {})
                    for key, value in env_or_headers.items():
                        if env_special_value in value:
                            env_or_headers[key] = env_or_headers[key].replace(env_special_value, final_value)

            print()

        raw_api_key = config.get("apiKey")
        if isinstance(raw_api_key, str):
            substituted_api_key = raw_api_key
            for input_id, val in resolved_inputs.items():
                substituted_api_key = substituted_api_key.replace(f"${{input:{input_id}}}", val)
            config["apiKey"] = substituted_api_key
        # Main agent loop
        async with Agent(
            provider=config.get("provider"),  # type: ignore
            model=config.get("model"),
            base_url=config.get("endpointUrl"),  # type: ignore[arg-type]
            api_key=config.get("apiKey"),
            servers=servers,  # type: ignore[arg-type]
            prompt=prompt,
        ) as agent:
            await agent.load_tools()
            print(ANSI.bold(ANSI.blue("Agent loaded with {} tools:".format(len(agent.available_tools)))))
            for t in agent.available_tools:
                print(ANSI.blue(f" • {t.function.name}"))

            while True:
                abort_event.clear()

                # Check if we should exit
                if exit_event.is_set():
                    return

                try:
                    user_input = await _async_prompt(exit_event=exit_event)
                    first_sigint = True
                except EOFError:
                    print(ANSI.red("\nEOF received, exiting."), flush=True)
                    break
                except KeyboardInterrupt:
                    if not first_sigint and abort_event.is_set():
                        continue
                    else:
                        print(ANSI.red("\nKeyboard interrupt during input processing."), flush=True)
                        break

                try:
                    async for chunk in agent.run(user_input, abort_event=abort_event):
                        if abort_event.is_set() and not first_sigint:
                            break
                        if exit_event.is_set():
                            return

                        if hasattr(chunk, "choices"):
                            delta = chunk.choices[0].delta
                            if delta.content:
                                print(delta.content, end="", flush=True)
                            if delta.tool_calls:
                                for call in delta.tool_calls:
                                    if call.id:
                                        print(f"<Tool {call.id}>", end="")
                                    if call.function.name:
                                        print(f"{call.function.name}", end=" ")
                                    if call.function.arguments:
                                        print(f"{call.function.arguments}", end="")
                        else:
                            print(
                                ANSI.green(f"\n\nTool[{chunk.name}] {chunk.tool_call_id}\n{chunk.content}\n"),
                                flush=True,
                            )

                    print()

                except Exception as e:
                    tb_str = traceback.format_exc()
                    print(ANSI.red(f"\nError during agent run: {e}\n{tb_str}"), flush=True)
                    first_sigint = True  # Allow graceful interrupt for the next command

    except Exception as e:
        tb_str = traceback.format_exc()
        print(ANSI.red(f"\nAn unexpected error occurred: {e}\n{tb_str}"), flush=True)
        raise e

    finally:
        if sigint_registered_in_loop:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (AttributeError, NotImplementedError):
                pass
        else:
            signal.signal(signal.SIGINT, original_sigint_handler)


@app.command("run", help="Run the Agent in the CLI")
@click.argument("path", required=False)
def run(path: Optional[str]) -> None:
    """
    Run the agent from PATH: a local folder containing an agent.json file or a built-in agent
    stored in the 'tiny-agents/tiny-agents' Hugging Face dataset
    (https://huggingface.co/datasets/tiny-agents/tiny-agents).
    """
    try:
        asyncio.run(run_agent(path))
    except KeyboardInterrupt:
        print(ANSI.red("\nApplication terminated by KeyboardInterrupt."), flush=True)
        raise click.exceptions.Exit(code=130)
    except Exception as e:
        print(ANSI.red(f"\nAn unexpected error occurred: {e}"), flush=True)
        raise e


if __name__ == "__main__":
    app()
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
# WARNING
# This entire file has been adapted from the sync-client code in `src/huggingface_hub/inference/_client.py`.
# Any change in InferenceClient will be automatically reflected in AsyncInferenceClient.
# To re-generate the code, run `make style` or `python ./utils/generate_async_inference_client.py --update`.
# WARNING
import asyncio
import base64
import logging
import os
import re
import warnings
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, AsyncIterable, Literal, Optional, Union, overload

import httpx2

from huggingface_hub import constants
from huggingface_hub.errors import BadRequestError, HfHubHTTPError, InferenceTimeoutError
from huggingface_hub.inference._common import (
    TASKS_EXPECTING_IMAGES,
    ContentT,
    RequestParameters,
    _async_stream_chat_completion_response,
    _async_stream_text_generation_response,
    _b64_encode,
    _b64_to_image,
    _bytes_to_dict,
    _bytes_to_image,
    _bytes_to_list,
    _get_unsupported_text_generation_kwargs,
    _import_numpy,
    _set_unsupported_text_generation_kwargs,
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
    get_async_session,
    hf_raise_for_status,
    validate_hf_hub_args,
)

from .._common import _async_yield_from


if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

logger = logging.getLogger(__name__)


MODEL_KWARGS_NOT_USED_REGEX = re.compile(r"The following `model_kwargs` are not used by the model: \[(.*?)\]")


class AsyncInferenceClient:
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

        self.exit_stack = AsyncExitStack()
        self._async_client: Optional[httpx2.AsyncClient] = None

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        """Close the client.

        This method is automatically called when using the client as a context manager.
        """
        await self.exit_stack.aclose()

    async def _get_async_client(self):
        """Get a unique async client for this AsyncInferenceClient instance.

        Returns the same client instance on subsequent calls, ensuring proper
        connection reuse and resource management through the exit stack.
        """
        if self._async_client is None:
            self._async_client = await self.exit_stack.enter_async_context(get_async_session())
        return self._async_client

    @overload
    async def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[False] = ...
    ) -> bytes: ...

    @overload
    async def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[True] = ...
    ) -> AsyncIterable[str]: ...

    @overload
    async def _inner_post(
        self, request_parameters: RequestParameters, *, stream: bool = False
    ) -> bytes | AsyncIterable[str]: ...

    async def _inner_post(
        self, request_parameters: RequestParameters, *, stream: bool = False
    ) -> bytes | AsyncIterable[str]:
        """Make a request to the inference server."""

        # TODO: this should be handled in provider helpers directly
        if request_parameters.task in TASKS_EXPECTING_IMAGES and "Accept" not in request_parameters.headers:
            request_parameters.headers["Accept"] = "image/png"

        try:
            client = await self._get_async_client()
            if stream:
                response = await self.exit_stack.enter_async_context(
                    client.stream(
                        "POST",
                        request_parameters.url,
                        json=request_parameters.json,
                        data=request_parameters.data,
                        headers=request_parameters.headers,
                        cookies=self.cookies,
                        timeout=self.timeout,
                    )
                )
                hf_raise_for_status(response)
                return _async_yield_from(client, response)
            else:
                response = await client.post(
                    request_parameters.url,
                    json=request_parameters.json,
                    data=request_parameters.data,
                    headers=request_parameters.headers,
                    cookies=self.cookies,
                    timeout=self.timeout,
                )
                hf_raise_for_status(response)
                return response.content
        except asyncio.TimeoutError as error:
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

    async def audio_classification(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.audio_classification("audio.flac")
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
        response = await self._inner_post(request_parameters)
        return AudioClassificationOutputElement.parse_obj_as_list(response)

    async def audio_to_audio(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> audio_output = await client.audio_to_audio("audio.flac")
        >>> async for i, item in enumerate(audio_output):
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
        response = await self._inner_post(request_parameters)
        audio_output = AudioToAudioOutputElement.parse_obj_as_list(response)
        for item in audio_output:
            item.blob = base64.b64decode(item.blob)
        return audio_output

    async def automatic_speech_recognition(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.automatic_speech_recognition("hello_world.flac").text
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
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_params=request_parameters)
        return AutomaticSpeechRecognitionOutput.parse_obj_as_instance(response)

    @overload
    async def chat_completion(  # type: ignore
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
    async def chat_completion(  # type: ignore
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
    ) -> AsyncIterable[ChatCompletionStreamOutput]: ...

    @overload
    async def chat_completion(
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
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]: ...

    async def chat_completion(
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
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]:
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> await client.chat_completion(messages, max_tokens=100)
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> async for token in await client.chat_completion(messages, max_tokens=10, stream=True):
        ...     print(token)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        (...)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ```

        Example using OpenAI's syntax:
        ```py
        # Must be run in an async context
        # instead of `from openai import OpenAI`
        from huggingface_hub import AsyncInferenceClient

        # instead of `client = OpenAI(...)`
        client = AsyncInferenceClient(
            base_url=...,
            api_key=...,
        )

        output = await client.chat.completions.create(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient

        # provide a remote URL
        >>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        # or a base64-encoded image
        >>> image_path = "/path/to/image.jpeg"
        >>> with open(image_path, "rb") as f:
        ...     base64_image = base64.b64encode(f.read()).decode("utf-8")
        >>> image_url = f"data:image/jpeg;base64,{base64_image}"

        >>> client = AsyncInferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
        >>> output = await client.chat.completions.create(
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
        # Must be run in an async context
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
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

        >>> response = await client.chat_completion(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
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
        >>> response = await client.chat_completion(
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
        data = await self._inner_post(request_parameters, stream=stream)

        if stream:
            return _async_stream_chat_completion_response(data)  # type: ignore

        return ChatCompletionOutput.parse_obj_as_instance(data)  # type: ignore

    async def document_question_answering(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
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
        response = await self._inner_post(request_parameters)
        return DocumentQuestionAnsweringOutputElement.parse_obj_as_list(response)

    async def feature_extraction(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.feature_extraction("Hi, who are you?")
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
        response = await self._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(provider_helper.get_response(response), dtype="float32")

    async def fill_mask(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.fill_mask("The goal of life is <mask>.")
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
        response = await self._inner_post(request_parameters)
        return FillMaskOutputElement.parse_obj_as_list(response)

    async def image_classification(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
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
        response = await self._inner_post(request_parameters)
        return ImageClassificationOutputElement.parse_obj_as_list(response)

    async def image_segmentation(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_segmentation("cat.jpg")
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
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        output = ImageSegmentationOutputElement.parse_obj_as_list(response)
        for item in output:
            item.mask = _b64_to_image(item.mask)  # type: ignore
        return output

    async def image_to_image(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> image = await client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
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
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return _bytes_to_image(response)

    async def image_to_video(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> video = await client.image_to_video("cat.jpg", model="Wan-AI/Wan2.2-I2V-A14B", prompt="turn the cat into a tiger")
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
        response = await self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return response

    async def image_to_text(self, image: ContentT, *, model: str | None = None) -> ImageToTextOutput:
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_to_text("cat.jpg")
        'a cat standing in a grassy field '
        >>> await client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
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
        response = await self._inner_post(request_parameters)
        output_list: list[ImageToTextOutput] = ImageToTextOutput.parse_obj_as_list(response)
        return output_list[0]

    async def object_detection(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.object_detection("people.jpg")
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
        response = await self._inner_post(request_parameters)
        return ObjectDetectionOutputElement.parse_obj_as_list(response)

    async def question_answering(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
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
        response = await self._inner_post(request_parameters)
        # Parse the response as a single `QuestionAnsweringOutputElement` when top_k is 1 or not provided, or a list of `QuestionAnsweringOutputElement` to ensure backward compatibility.
        output = QuestionAnsweringOutputElement.parse_obj(response)
        return output

    async def sentence_similarity(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.sentence_similarity(
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
        response = await self._inner_post(request_parameters)
        return _bytes_to_list(response)

    async def summarization(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.summarization("The Eiffel tower...")
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
        response = await self._inner_post(request_parameters)
        return SummarizationOutput.parse_obj_as_list(response)[0]

    async def table_question_answering(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> query = "How many stars does the transformers repository have?"
        >>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
        >>> await client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
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
        response = await self._inner_post(request_parameters)
        return TableQuestionAnsweringOutputElement.parse_obj_as_instance(response)

    async def tabular_classification(self, table: dict[str, Any], *, model: str | None = None) -> list[str]:
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
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
        >>> await client.tabular_classification(table=table, model="julien-c/wine-quality")
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
        response = await self._inner_post(request_parameters)
        return _bytes_to_list(response)

    async def tabular_regression(self, table: dict[str, Any], *, model: str | None = None) -> list[float]:
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> table = {
        ...     "Height": ["11.52", "12.48", "12.3778"],
        ...     "Length1": ["23.2", "24", "23.9"],
        ...     "Length2": ["25.4", "26.3", "26.5"],
        ...     "Length3": ["30", "31.2", "31.1"],
        ...     "Species": ["Bream", "Bream", "Bream"],
        ...     "Width": ["4.02", "4.3056", "4.6961"],
        ... }
        >>> await client.tabular_regression(table, model="scikit-learn/Fish-Weight")
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
        response = await self._inner_post(request_parameters)
        return _bytes_to_list(response)

    async def text_classification(
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
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.text_classification("I like you")
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
