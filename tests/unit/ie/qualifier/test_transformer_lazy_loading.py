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
