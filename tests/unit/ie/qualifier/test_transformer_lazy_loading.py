from types import SimpleNamespace

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
    def __init__(self):
        super().__init__({"input_ids": torch.tensor([[101, 102]])})
        self.encodings = [SimpleNamespace(offsets=[(0, 1)])]

    def char_to_token(self, _char_i: int) -> int:
        return 0


class _FakeTokenizer:
    def __call__(self, _text: str, return_tensors: str):
        assert return_tensors == "pt"
        return _FakeInputs()


class _FakeModel:
    config = SimpleNamespace(
        id2label={
            0: "B-GIVENNAME",
            1: "B-SURNAME",
            2: "B-DATE",
        }
    )

    @staticmethod
    def forward(_input_ids):
        # DATE has the highest single-label probability. NAME has two mapped
        # labels, so summing would bias toward NAME, but mean should keep DATE.
        return SimpleNamespace(logits=torch.tensor([[[2.0, 1.9, 2.1]]]))


class _DummyMappedTransformer(QualifierTransformer):
    def __init__(self):
        self._dummy_tokenizer = _FakeTokenizer()
        self._dummy_model = _FakeModel()
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

    def _detect_qualifiers(self, doc: Doc) -> None:
        return None


def test_predict_multiclass_reduction_uses_mean_to_avoid_size_bias():
    """Average reduced-label probabilities to avoid class-size bias."""

    # Arrange
    transformer = _DummyMappedTransformer()

    # Act
    qualifier, probability = transformer._predict_multiclass(
        "x",
        0,
        1,
        qualifier_mapping={
            "GIVENNAME": "NAME",
            "SURNAME": "NAME",
        },
    )

    # Assert
    assert qualifier == "DATE"
    assert 0 < probability < 1
