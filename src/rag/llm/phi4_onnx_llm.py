import onnxruntime_genai as og

from llama_index.core.llms.llm import LLM
from llama_index.core.llms import (
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import Field
from pydantic import PrivateAttr


class Phi4OnnxLLM(LLM):
    """
    Custom LlamaIndex LLM wrapper for Phi-4-mini-instruct-onnx
    using ONNX Runtime GenAI 0.2.x API for CPU inference.
    """

    model_dir: str = Field(description="Path to ONNX model directory")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)

    _model: og.Model = PrivateAttr()
    _tokenizer: og.Tokenizer = PrivateAttr()
    _generator: og.Generator = PrivateAttr()
    _generator_params: og.GeneratorParams = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load ONNX model + tokenizer
        self._model = og.Model(self.model_dir)
        self._tokenizer = og.Tokenizer(self._model)

        # Build generator params (ORT GenAI 0.2.x API)
        params = og.GeneratorParams(self._model)

        # Old API: set_search_options(**kwargs)
        params.set_search_options(
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            max_length=int(self.max_new_tokens),   # THIS is the correct API
        )

        self._generator_params = params
        self._generator = og.Generator(self._model, params)

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
        input_tokens = self._tokenizer.encode(prompt)
        output_tokens = self._generator.generate(input_tokens)
        text = self._tokenizer.decode(output_tokens)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        input_tokens = self._tokenizer.encode(prompt)

        def gen():
            for token in self._generator.generate_stream(input_tokens):
                yield self._tokenizer.decode([token])

        return CompletionResponseGen(gen())

    # -----------------------------
    # Required abstract methods
    # -----------------------------
    async def acomplete(self, prompt: str, **kwargs):
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        return self.stream_complete(prompt, **kwargs)

    def chat(self, messages, **kwargs):
        return self.complete(messages[-1].content, **kwargs)

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        return self.stream_complete(messages[-1].content, **kwargs)

    async def astream_chat(self, messages, **kwargs):
        return self.stream_chat(messages, **kwargs)
