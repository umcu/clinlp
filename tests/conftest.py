import pytest
import spacy

import clinlp  # noqa F401
from clinlp.ie import SPANS_KEY


# Arrange
@pytest.fixture(scope="class")
def nlp():
    return spacy.blank("clinlp")


# Arrange
@pytest.fixture(scope="class")
def nlp_entity(nlp):
    nlp.add_pipe("clinlp_normalizer")

    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    return nlp
