"""Pluggable answer-generation models.

The Streamlit app and AnswerService only know the tiny `LLMClient` protocol, so
GPT-4.1, a base Qwen model or a fine-tuned Qwen adapter are swapped purely by
configuration (see src/config.py: LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL,
QWEN_MODEL_NAME, QWEN_ADAPTER_PATH). Nothing is downloaded or loaded until the
LLM is built, and building happens lazily on the first question.
"""
import logging
import os
from typing import Optional, Protocol

from src import config

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIChatLLM:
    """OpenAI (gpt-4.1, as already used by RAGSearch) or any OpenAI-compatible server (e.g. Qwen on vLLM)."""

    def __init__(self, model: str, base_url: str = ""):
        from langchain_openai import ChatOpenAI  # lazy

        if not base_url and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing from environment variables.")
        kwargs = {"model": model, "temperature": 0}
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "not-needed")
        self._llm = ChatOpenAI(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return str(self._llm.invoke([("system", system_prompt), ("human", user_prompt)]).content)


class QwenLocalLLM:
    """Hugging Face Qwen in-process, with an optional LoRA adapter from a later fine-tuning phase."""

    def __init__(self, model_name: str, adapter_path: str = "", max_new_tokens: int = 400):
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        if adapter_path:
            from peft import PeftModel  # only needed when an adapter is configured

            self._model = PeftModel.from_pretrained(self._model, adapter_path)
        self._model.eval()
        self._torch = torch

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        ids = self._tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()


def build_llm(provider: Optional[str] = None) -> Optional[LLMClient]:
    """Create the configured LLM, or None for provider 'none'. Raises if misconfigured."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "none":
        return None
    if provider == "openai":
        return OpenAIChatLLM(config.LLM_MODEL)
    if provider == "openai_compatible":
        if not config.LLM_BASE_URL:
            raise RuntimeError("LLM_BASE_URL is required for LLM_PROVIDER=openai_compatible.")
        return OpenAIChatLLM(config.LLM_MODEL, base_url=config.LLM_BASE_URL)
    if provider == "qwen_local":
        return QwenLocalLLM(config.QWEN_MODEL_NAME, config.QWEN_ADAPTER_PATH)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'.")
