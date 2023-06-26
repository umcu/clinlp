import pytest
import spacy
from spacy.tokens import Span

import clinlp
import clinlp.qualifier
from clinlp.qualifier.qualifier import QUALIFIERS_ATTR, Qualifier


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


class TestUnitQualifier:
    def test_qualifier(self):
        q = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"])

        assert q["AFFIRMED"]
        assert q["NEGATED"]

    def test_spacy_has_extension(self):
        assert Span.has_extension(QUALIFIERS_ATTR)

    def test_spacy_extension_default(self, nlp):
        doc = nlp("dit is een test")
        assert getattr(doc[0:3]._, QUALIFIERS_ATTR) is None
