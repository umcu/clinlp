from types import SimpleNamespace

from click.testing import CliRunner

from clinlp.cli import cli


def test_install_transformers_requires_optional_dependency(monkeypatch):
    """Fail with a clear message when transformers is not installed."""

    # Arrange
    def _mock_find_spec(module_name: str):
        if module_name == "transformers":
            return None

        return object()

    monkeypatch.setattr("clinlp.cli.importlib.util.find_spec", _mock_find_spec)

    runner = CliRunner()

    # Act
    result = runner.invoke(cli, ["install", "transformers"])

    # Assert
    assert result.exit_code != 0
    assert 'pip install "clinlp[transformers]"' in result.output


def test_install_transformers_prefetches_all_models(monkeypatch):
    """Prefetch all configured transformer models."""

    # Arrange
    class _NegationTransformer:
        PRETRAINED_MODEL_NAME_OR_PATH = "umcu/negation"
        REVISION = "rev-a"
        prefetch_force_download: bool | None = None
        prefetch_count = 0

        @classmethod
        def prefetch(cls, *, force_download: bool = False) -> None:
            cls.prefetch_force_download = force_download
            cls.prefetch_count += 1

    class _ExperiencerTransformer:
        PRETRAINED_MODEL_NAME_OR_PATH = "umcu/experiencer"
        REVISION = "rev-b"
        prefetch_force_download: bool | None = None
        prefetch_count = 0

        @classmethod
        def prefetch(cls, *, force_download: bool = False) -> None:
            cls.prefetch_force_download = force_download
            cls.prefetch_count += 1

    class _PiiTransformer:
        PRETRAINED_MODEL_NAME_OR_PATH = "umcu/pii"
        REVISION = "rev-c"
        prefetch_force_download: bool | None = None
        prefetch_count = 0

        @classmethod
        def prefetch(cls, *, force_download: bool = False) -> None:
            cls.prefetch_force_download = force_download
            cls.prefetch_count += 1

    class _CardioTransformer:
        PRETRAINED_MODEL_NAME_OR_PATH = "dt4h/cardio"
        REVISION = "rev-d"
        prefetch_force_download: bool | None = None
        prefetch_count = 0

        @classmethod
        def prefetch(cls, *, force_download: bool = False) -> None:
            cls.prefetch_force_download = force_download
            cls.prefetch_count += 1

    fake_transformer_module = SimpleNamespace(
        NegationTransformer=_NegationTransformer,
        ExperiencerTransformer=_ExperiencerTransformer,
        PiiTransformer=_PiiTransformer,
        CardioTransformer=_CardioTransformer,
    )

    def _mock_find_spec(module_name: str):
        if module_name == "transformers":
            return object()

        return None

    def _mock_import_module(module_name: str):
        if module_name == "clinlp.ie.qualifier.transformer":
            return fake_transformer_module

        msg = f"Unexpected import requested in test: {module_name}"
        raise AssertionError(msg)

    monkeypatch.setattr("clinlp.cli.importlib.util.find_spec", _mock_find_spec)
    monkeypatch.setattr("clinlp.cli.importlib.import_module", _mock_import_module)

    runner = CliRunner()

    # Act
    result = runner.invoke(cli, ["install", "transformers", "--force-download"])

    # Assert
    assert result.exit_code == 0
    assert _NegationTransformer.prefetch_force_download is True
    assert _ExperiencerTransformer.prefetch_force_download is True
    assert _PiiTransformer.prefetch_force_download is True
    assert _CardioTransformer.prefetch_force_download is True
    assert _NegationTransformer.prefetch_count == 1
    assert _ExperiencerTransformer.prefetch_count == 1
    assert _PiiTransformer.prefetch_count == 1
    assert _CardioTransformer.prefetch_count == 1
    assert "Done prefetching transformer models." in result.output
