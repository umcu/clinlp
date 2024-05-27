import pytest
import spacy

import clinlp  # noqa: F401
from clinlp.exceptions import VersionMismatchWarning


class TestUnitCreateModel:
    def test_apply_model(self):
        # Arrange
        nlp = spacy.blank("clinlp")

        # Act
        doc = nlp("dit is een test")

        # Assert
        assert len(doc) == 4

    def test_version(self):
        # Arrange & Act
        nlp = spacy.blank("clinlp")

        # Assert
        assert "clinlp_version" in nlp.meta

    def test_load_wrong_version(self):
        # Assert
        with pytest.warns(VersionMismatchWarning):
            # Act
            _ = spacy.load("tests/data/test_spacy_model")
