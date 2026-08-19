"""Local Gemma answer generation through Hugging Face Transformers."""

from dataclasses import dataclass, field
from typing import Any


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"


@dataclass
class GemmaAnswerModel:
    """Generate answers with an instruction-tuned Gemma model."""

    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 512
    _processor: Any | None = field(
        default=None,
        repr=False,
    )
    _model: Any | None = field(default=None, repr=False)

    def generate(self, prompt: str) -> str:
        """Generate one answer from the supplied grounded prompt."""
        processor, model = self._get_components()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        response = processor.decode(
            outputs[0][input_length:],
            skip_special_tokens=False,
        )
        parsed_response = processor.parse_response(
            response,
            prefix=inputs["input_ids"],
        )
        content = parsed_response.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Gemma returned an unexpected output format.")

        return content.strip()

    def _get_components(self) -> tuple[Any, Any]:
        """Load and cache the processor and model on first use."""
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype="auto",
        )
        return self._processor, self._model
