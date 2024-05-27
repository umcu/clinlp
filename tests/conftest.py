from pathlib import Path

import pytest
import spacy
from spacy import Language

import clinlp  # noqa F401
from clinlp.ie import SPANS_KEY

TEST_DATA_DIR = Path("tests/test_data")


class MockToken:
    def __init__(self, text: str):
        self.text = text
        self.is_sent_start = False


def get_mock_tokens(texts: list[str]):
    return [MockToken(text) for text in texts]


def _make_nlp():
    return spacy.blank("clinlp")


def _make_nlp_entity(nlp: Language):
    nlp.add_pipe("clinlp_normalizer")

    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    return nlp


# Arrange
@pytest.fixture
def nlp():
    return _make_nlp()


# Arrange
@pytest.fixture
def nlp_entity(nlp):
    return _make_nlp_entity(nlp)
