import onnxruntime_genai as og
from typing import Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.llms import (
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import Field


class Phi4OnnxLLM(LLM):
    """
    Custom LlamaIndex LLM wrapper for Phi-4-mini-instruct-onnx
    using ONNX Runtime GenAI for CPU inference.
    """

    # Pydantic model fields (REQUIRED)
    model_dir: str = Field(description="Path to ONNX model directory")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load ONNX model + tokenizer
        self.tokenizer = og.Tokenizer(self.model_dir)
        self.model = og.Model(self.model_dir)
        self.generator = og.Generator(self.model, self.tokenizer)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_new_tokens,
            model_name="phi-4-mini-instruct-onnx",
        )

    # -----------------------------
    # Core completion methods
    # -----------------------------
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        inputs = self.tokenizer.encode(prompt)
        outputs = self.generator.generate(
            inputs,
            max_length=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        text = self.tokenizer.decode(outputs)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        inputs = self.tokenizer.encode(prompt)

        def gen():
            for token in self.generator.generate_stream(
                inputs,
                max_length=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ):
                yield self.tokenizer.decode([token])

        return CompletionResponseGen(gen())

    # -----------------------------
    # Required abstract methods
    # -----------------------------
    async def acomplete(self, prompt: str, **kwargs):
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        return self.stream_complete(prompt, **kwargs)

    def chat(self, messages, **kwargs):
        prompt = messages[-1].content
        return self.complete(prompt, **kwargs)

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        prompt = messages[-1].content
        return self.stream_complete(prompt, **kwargs)

    async def astream_chat(self, messages, **kwargs):
        return self.stream_chat(messages, **kwargs)
