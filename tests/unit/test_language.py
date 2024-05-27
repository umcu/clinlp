import pytest
import spacy
from tests.conftest import TEST_DATA_DIR

from clinlp.exceptions import VersionMismatchWarning


class TestUnitCreateModel:
    def test_apply_model(self, nlp):
        # Act
        doc = nlp("dit is een test")

        # Assert
        assert len(doc) == 4

    def test_version(self, nlp):
        # Assert

        assert "clinlp_version" in nlp.meta

    @pytest.mark.filterwarnings("ignore:.*W095.*:UserWarning")
    def test_load_wrong_version(self):
        # Assert
        with pytest.warns(VersionMismatchWarning):
            # Act
            _ = spacy.load(TEST_DATA_DIR / "test_spacy_model")
