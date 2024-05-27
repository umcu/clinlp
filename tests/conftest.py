import pytest
import spacy
from spacy import Language

import clinlp  # noqa F401
from clinlp.ie import SPANS_KEY


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
