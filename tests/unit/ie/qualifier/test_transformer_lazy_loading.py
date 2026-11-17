from types import SimpleNamespace

import pytest
import torch
from spacy.tokens import Doc

from clinlp.ie.qualifier import QualifierTransformer


class DummyQualifierTransformer(QualifierTransformer):
    """Minimal concrete transformer for testing lazy loading."""

    PRETRAINED_MODEL_NAME_OR_PATH = "dummy/model"
    REVISION = "dummy-revision"

    @property
    def qualifier_classes(self) -> dict:
        """No qualifier classes required for these tests."""
        return {}

    def _detect_qualifiers(self, doc: Doc) -> None:
        """No-op detector implementation for tests."""


def test_transformer_assets_are_loaded_lazily(monkeypatch):
    """Do not load tokenizer/model until the corresponding property is accessed."""

    # Arrange
    DummyQualifierTransformer._tokenizer = None
    DummyQualifierTransformer._model = None

    tokenizer_calls = []
    model_calls = []

    def _mock_tokenizer_from_pretrained(**kwargs):
        tokenizer_calls.append(kwargs)
        return object()

    def _mock_model_from_pretrained(**kwargs):
        model_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoTokenizer.from_pretrained",
        _mock_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoModelForTokenClassification.from_pretrained",
        _mock_model_from_pretrained,
    )

    transformer = DummyQualifierTransformer()

    # Act
    _ = transformer.tokenizer
    _ = transformer.model
    _ = transformer.tokenizer
    _ = transformer.model

    # Assert
    assert len(tokenizer_calls) == 1
    assert len(model_calls) == 1


def test_prefetch_passes_force_download(monkeypatch):
    """Forward force_download to Hugging Face loaders."""

    # Arrange
    DummyQualifierTransformer._tokenizer = None
    DummyQualifierTransformer._model = None

    tokenizer_calls = []
    model_calls = []

    def _mock_tokenizer_from_pretrained(**kwargs):
        tokenizer_calls.append(kwargs)
        return object()

    def _mock_model_from_pretrained(**kwargs):
        model_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoTokenizer.from_pretrained",
        _mock_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoModelForTokenClassification.from_pretrained",
        _mock_model_from_pretrained,
    )

    # Act
    DummyQualifierTransformer.prefetch(force_download=True)

    # Assert
    assert tokenizer_calls[0]["force_download"] is True
    assert model_calls[0]["force_download"] is True
    assert tokenizer_calls[0]["trust_remote_code"] is False
    assert model_calls[0]["trust_remote_code"] is False


class DummyTrustedQualifierTransformer(DummyQualifierTransformer):
    TRUST_REMOTE_CODE = True


def test_prefetch_passes_trust_remote_code(monkeypatch):
    """Forward trust_remote_code for models requiring custom remote code."""

    # Arrange
    DummyTrustedQualifierTransformer._tokenizer = None
    DummyTrustedQualifierTransformer._model = None

    tokenizer_calls = []
    model_calls = []

    def _mock_tokenizer_from_pretrained(**kwargs):
        tokenizer_calls.append(kwargs)
        return object()

    def _mock_model_from_pretrained(**kwargs):
        model_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoTokenizer.from_pretrained",
        _mock_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "clinlp.ie.qualifier.transformer.AutoModelForTokenClassification.from_pretrained",
        _mock_model_from_pretrained,
    )

    # Act
    DummyTrustedQualifierTransformer.prefetch()

    # Assert
    assert tokenizer_calls[0]["trust_remote_code"] is True
    assert model_calls[0]["trust_remote_code"] is True


class _FakeInputs(dict):
    """One transformer token per character."""

    def __init__(self, text: str):
        super().__init__({"input_ids": torch.zeros((1, len(text)), dtype=torch.long)})
        self.encodings = [
            SimpleNamespace(offsets=[(i, i + 1) for i in range(len(text))])
        ]

    def char_to_token(self, char_i: int) -> int:
        return char_i


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str):
        assert return_tensors == "pt"
        return _FakeInputs(text)


class _FakeModel:
    """Model whose softmax output equals the given per-token probabilities."""

    def __init__(self, labels: list[str], token_probs: list[list[float]]):
        self.config = SimpleNamespace(id2label=dict(enumerate(labels)))
        self._logits = torch.log(torch.tensor([token_probs]))

    def forward(self, _input_ids):
        return SimpleNamespace(logits=self._logits)


class _DummyMappedTransformer(QualifierTransformer):
    def __init__(self, labels: list[str], token_probs: list[list[float]]):
        self._dummy_tokenizer = _FakeTokenizer()
        self._dummy_model = _FakeModel(labels, token_probs)
        super().__init__()

    @property
    def tokenizer(self):
        return self._dummy_tokenizer

    @property
    def model(self):
        return self._dummy_model

    @property
    def qualifier_classes(self) -> dict:
        return {}

    def _detect_qualifiers(self, _doc: Doc) -> None:
        return None


def test_predict_multiclass_sums_mapped_labels_and_averages_over_tokens():
    # Arrange
    transformer = _DummyMappedTransformer(
        labels=["O", "B-GIVENNAME", "I-GIVENNAME", "B-SURNAME", "B-DATE"],
        token_probs=[
            [0.1, 0.5, 0.1, 0.1, 0.2],  # NAME: 0.7, DATE: 0.2
            [0.1, 0.1, 0.1, 0.4, 0.3],  # NAME: 0.6, DATE: 0.3
        ],
    )

    # Act
    qualifier, probability = transformer._predict_multiclass(
        "xy",
        0,
        2,
        qualifier_mapping={
            "GIVENNAME": "NAME",
            "SURNAME": "NAME",
        },
    )

    # Assert
    assert qualifier == "NAME"
    assert probability == pytest.approx((0.7 + 0.6) / 2)


@pytest.mark.parametrize("qualifier_mapping", [None, {}])
def test_predict_multiclass_merges_bio_prefixes_without_mapping(qualifier_mapping):
    # Arrange
    transformer = _DummyMappedTransformer(
        labels=["O", "B-SYMPTOM", "I-SYMPTOM", "B-DISEASE"],
        token_probs=[
            [0.1, 0.6, 0.1, 0.2],  # SYMPTOM: 0.7
            [0.1, 0.1, 0.6, 0.2],  # SYMPTOM: 0.7
        ],
    )

    # Act
    qualifier, probability = transformer._predict_multiclass(
        "xy", 0, 2, qualifier_mapping=qualifier_mapping
    )

    # Assert
    assert qualifier == "SYMPTOM"
    assert probability == pytest.approx(0.7)
