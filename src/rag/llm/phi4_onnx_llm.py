import os
from typing import Optional, List

from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen
from llama_index.core.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field

import onnxruntime_genai as og


class Phi4OnnxLLM(LLM):
    """
    Custom LlamaIndex LLM wrapper for Phi-4-mini-instruct-onnx
    using ONNX Runtime GenAI for CPU inference.
    """

    model_dir: str = Field(description="Path to ONNX model directory")
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)

        # Load ONNX model + tokenizer
        self.tokenizer = og.Tokenizer(model_dir)
        self.model = og.Model(model_dir)
        self.generator = og.Generator(self.model, self.tokenizer)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_new_tokens,
            model_name="phi-4-mini-instruct-onnx",
        )

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
