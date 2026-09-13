from __future__ import annotations

import time
import warnings
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 256
    seed: int | None = None
    deterministic: bool = False


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float


def _extract_images_and_text(messages: list[dict]) -> tuple[list[dict], list[Any]]:
    images: list[Any] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(part.get("image"))
    return messages, images


def _normalize_content_for_multimodal_template(messages: list[dict]) -> list[dict]:
    """Ensure every message's ``content`` is list-of-typed-parts form.

    Some multimodal chat templates (e.g. SmolVLM/Idefics-family) only know how to
    iterate typed content parts (``[{"type": "text", "text": ...}]``) and silently
    drop plain-string content — producing a prompt with the user's text missing
    entirely rather than an error. Normalizing here makes text-only calls behave
    the same regardless of which multimodal model is loaded.
    """
    normalized: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            normalized.append({**msg, "content": [{"type": "text", "text": content}]})
        else:
            normalized.append(msg)
    return normalized


@runtime_checkable
class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def load(self) -> None: ...

    def generate(self, messages: list[dict], config: GenerationConfig) -> GenerationResult: ...

    def apply_chat_template(self, messages: list[dict]) -> str: ...


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}


class HFEngine:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        dtype: str = "bf16",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_multimodal = False
        self._loaded = False

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self._processor = None
        self._is_multimodal = False
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name, revision=self.revision)
            self._is_multimodal = getattr(self._processor, "image_processor", None) is not None
        except Exception as exc:
            # Falling back to text-only silently would be actively wrong for a model
            # that genuinely is multimodal but failed to load its processor for some
            # other reason (e.g. a missing optional dependency) — surface it loudly
            # rather than hiding a broken vision path behind a quiet degrade.
            import warnings

            warnings.warn(
                f"AutoProcessor.from_pretrained failed for {self.model_name!r}, "
                f"falling back to text-only tokenizer: {exc!r}",
                stacklevel=2,
            )
            self._processor = None

        if self._processor is not None:
            self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)

        torch_dtype_name = _DTYPE_ALIASES.get(self.dtype, self.dtype)
        model_cls = AutoModelForCausalLM
        if self._is_multimodal:
            try:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText
            except ImportError:
                pass

        self._model = model_cls.from_pretrained(
            self.model_name,
            revision=self.revision,
            dtype=getattr(torch, torch_dtype_name),
            device_map=self.device,
        )
        self._loaded = True

    def apply_chat_template(self, messages: list[dict]) -> str:
        if self._is_multimodal and self._processor is not None:
            messages = _normalize_content_for_multimodal_template(messages)
            return self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if self._tokenizer is not None:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def generate(self, messages: list[dict], config: GenerationConfig) -> GenerationResult:
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("load() must be called before generate()")

        import torch

        if config.seed is not None:
            torch.manual_seed(config.seed)

        deterministic_ctx = None
        if config.deterministic:
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            deterministic_ctx = torch.use_deterministic_algorithms(True)

        messages, images = _extract_images_and_text(messages)

        try:
            if images and self._is_multimodal and self._processor is not None:
                prompt = self.apply_chat_template(messages)
                inputs = self._processor(text=prompt, images=images, return_tensors="pt")
            else:
                prompt = self.apply_chat_template(messages)
                inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]

            do_sample = config.temperature > 0 and not config.deterministic

            ttft_recorded = False
            ttft_ms = 0.0
            start_time = time.perf_counter()

            first_forward = True
            total_completion_tokens = 0

            class FirstTokenCallback:
                def __init__(self, parent):
                    self.parent = parent

                def __call__(self, input_ids, scores, **kwargs):
                    nonlocal ttft_recorded, ttft_ms, first_forward, total_completion_tokens
                    if first_forward:
                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                        ttft_recorded = True
                        first_forward = False
                    total_completion_tokens = input_ids.shape[1] - prompt_tokens

            from transformers import StoppingCriteria, StoppingCriteriaList

            class _TTFTStoppingCriteria(StoppingCriteria):
                def __init__(self, callback):
                    self.callback = callback

                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    self.callback(input_ids, scores, **kwargs)
                    return False

            stopping_criteria = StoppingCriteriaList(
                [_TTFTStoppingCriteria(FirstTokenCallback(self))]
            )

            gen_kwargs: dict = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": do_sample,
                "temperature": config.temperature if do_sample else None,
                "top_p": config.top_p if do_sample else None,
                "top_k": config.top_k if do_sample else None,
                "repetition_penalty": config.repetition_penalty,
                "pad_token_id": self._tokenizer.eos_token_id,
                "stopping_criteria": stopping_criteria,
            }

            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - start_time) * 1000.0

            total_ms = (time.perf_counter() - start_time) * 1000.0
            completion_tokens = output_ids.shape[1] - prompt_tokens

            generated_text = self._tokenizer.decode(
                output_ids[0, prompt_tokens:], skip_special_tokens=True
            )

            tokens_per_sec = completion_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

            return GenerationResult(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                tokens_per_sec=tokens_per_sec,
            )
        finally:
            if deterministic_ctx is not None:
                deterministic_ctx.__exit__(None, None, None)


class LlamaCppEngine:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._loaded = False

    @property
    def name(self) -> str:
        return f"llama-cpp:{self.model_path}"

    def load(self) -> None:
        import os

        from llama_cpp import Llama

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._loaded = True

    def apply_chat_template(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def generate(self, messages: list[dict], config: GenerationConfig) -> GenerationResult:
        if not self._loaded or self._llm is None:
            raise RuntimeError("load() must be called before generate()")

        import os

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        prompt_tokens = 0
        completion_tokens = 0
        ttft_ms = 0.0
        ttft_recorded = False

        start_time = time.perf_counter()

        completion = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if not config.deterministic else 0.0,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            seed=config.seed if config.seed is not None else -1,
            stream=True,
        )

        text_parts: list[str] = []
        for chunk in completion:
            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - start_time) * 1000.0
                ttft_recorded = True

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    text_parts.append(content)

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", len(text_parts))

        total_ms = (time.perf_counter() - start_time) * 1000.0

        if completion_tokens == 0:
            completion_tokens = len(text_parts)

        tokens_per_sec = completion_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

        return GenerationResult(
            text="".join(text_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tokens_per_sec,
        )
