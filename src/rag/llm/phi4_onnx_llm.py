import onnxruntime_genai as og

from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms import CompletionResponse
from llama_index.core.bridge.pydantic import Field
from pydantic import PrivateAttr


class Phi4OnnxLLM(CustomLLM):
    model_dir: str = Field(description="Path to ONNX model directory")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)

    _model: og.Model = PrivateAttr()
    _tokenizer: og.Tokenizer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = og.Model(self.model_dir)
        self._tokenizer = og.Tokenizer(self._model)

    def _make_params(self, input_tokens):
        params = og.GeneratorParams(self._model)
        total_max = len(input_tokens) + int(self.max_new_tokens)
        params.set_search_options(
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            max_length=int(total_max),
        )
        return params

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_new_tokens,
            model_name="phi-4-mini-instruct-onnx",
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        input_tokens = self._tokenizer.encode(prompt)
        params = self._make_params(input_tokens)
        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)

        output_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if new_tokens:
                output_tokens.extend(new_tokens)
            if len(output_tokens) >= self.max_new_tokens:
                break

        text = self._tokenizer.decode(output_tokens)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs):
        input_tokens = self._tokenizer.encode(prompt)
        params = self._make_params(input_tokens)
        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)

        def gen():
            output_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if not new_tokens:
                    continue
                output_tokens.extend(new_tokens)
                yield self._tokenizer.decode(new_tokens)
                if len(output_tokens) >= self.max_new_tokens:
                    break

        return gen()

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
