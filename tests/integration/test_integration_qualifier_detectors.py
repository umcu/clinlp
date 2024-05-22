import pytest
import spacy

import clinlp  # noqa F401
from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ContextAlgorithm,
    ExperiencerTransformer,
    NegationTransformer,
)


@pytest.fixture
def nlp():
    nlp = spacy.blank("clinlp")

    nlp.add_pipe("clinlp_sentencizer")

    rbem = nlp.add_pipe("clinlp_rule_based_entity_matcher")
    rbem.load_concepts({"diabetes": ["diabetes"]})

    return nlp


@pytest.fixture
def text():
    return "The patient heeft geen diabetes in de familie."


class TestIntegrationQualifierDetector:
    def test_context_algorithm(self, nlp, text):
        doc = nlp(text)

        ca = ContextAlgorithm(nlp=nlp)

        doc = ca(doc)

        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Experiencer.Family" in doc.spans[SPANS_KEY][0]._.qualifiers_str

    def test_experiencer_transformer(self, nlp, text):
        doc = nlp(text)

        et = ExperiencerTransformer(nlp=nlp)

        doc = et(doc)

        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Experiencer.Family" in doc.spans[SPANS_KEY][0]._.qualifiers_str

    def test_negation_transformer(self, nlp, text):
        doc = nlp(text)

        et = NegationTransformer(nlp=nlp)

        doc = et(doc)

        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Presence.Absent" in doc.spans[SPANS_KEY][0]._.qualifiers_str
