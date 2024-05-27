import pytest
import spacy

import clinlp  # noqa F401


# Arrange
@pytest.fixture(scope="function")
def nlp():
    return spacy.blank("clinlp")
