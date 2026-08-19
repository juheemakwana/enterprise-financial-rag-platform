import pytest

from financial_rag.gemma import DEFAULT_MODEL_ID, GemmaAnswerModel


class FakeInputIds:
    shape = (1, 3)


class FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds())
        self.device = None

    def to(self, device: str) -> "FakeInputs":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self, content: object) -> None:
        self.content = content
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.decode_calls: list[tuple[object, bool]] = []
        self.parse_calls: list[tuple[str, object]] = []
        self.inputs = FakeInputs()

    def apply_chat_template(self, messages: object, **kwargs: object) -> FakeInputs:
        self.template_calls.append((messages, kwargs))
        return self.inputs

    def decode(self, tokens: object, skip_special_tokens: bool) -> str:
        self.decode_calls.append((tokens, skip_special_tokens))
        return "Revenue was $215.9 billion [1].<turn|>"

    def parse_response(self, response: str, prefix: object) -> dict[str, object]:
        self.parse_calls.append((response, prefix))
        return {"role": "assistant", "content": self.content}


class FakeModel:
    device = "mps"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        return [[10, 20, 30, 40]]


def test_gemma_answer_model_formats_prompt_and_returns_clean_text() -> None:
    processor = FakeProcessor("Revenue was $215.9 billion [1].")
    fake_model = FakeModel()
    model = GemmaAnswerModel(
        max_new_tokens=100,
        _processor=processor,
        _model=fake_model,
    )

    answer = model.generate("Answer using only this evidence.")

    assert answer == "Revenue was $215.9 billion [1]."
    assert len(processor.template_calls) == 1
    messages, template_arguments = processor.template_calls[0]
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer using only this evidence."}
            ],
        }
    ]
    assert template_arguments == {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert processor.inputs.device == "mps"
    assert fake_model.generate_calls == [
        {
            "input_ids": processor.inputs["input_ids"],
            "max_new_tokens": 100,
            "do_sample": False,
        }
    ]
    assert processor.decode_calls == [([40], False)]
    assert processor.parse_calls == [
        (
            "Revenue was $215.9 billion [1].<turn|>",
            processor.inputs["input_ids"],
        )
    ]


def test_gemma_answer_model_uses_official_instruction_model_by_default() -> None:
    model = GemmaAnswerModel(
        _processor=FakeProcessor("answer"),
        _model=FakeModel(),
    )

    assert model.model_id == DEFAULT_MODEL_ID
    assert model.model_id.endswith("-it")


@pytest.mark.parametrize(
    "content",
    [
        None,
        ["unexpected", "list"],
    ],
)
def test_gemma_answer_model_rejects_unexpected_output(content: object) -> None:
    model = GemmaAnswerModel(
        _processor=FakeProcessor(content),
        _model=FakeModel(),
    )

    with pytest.raises(RuntimeError, match="Gemma returned"):
        model.generate("Question and evidence")
