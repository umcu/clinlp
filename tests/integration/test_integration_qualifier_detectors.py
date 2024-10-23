import pytest
from tests.conftest import _make_nlp

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ContextAlgorithm,
    ExperiencerTransformer,
    NegationTransformer,
)


# Arrange
@pytest.fixture
def text():
    return "The patient heeft geen diabetes in de familie."


# Arrange
@pytest.fixture
def nlp_qualifier():
    nlp_qualifier = _make_nlp()

    nlp_qualifier.add_pipe("clinlp_sentencizer")

    rbem = nlp_qualifier.add_pipe("clinlp_rule_based_entity_matcher")
    rbem.add_term(concept="diabetes", term="diabetes")

    return nlp_qualifier


class TestIntegrationQualifierDetector:
    def test_context_algorithm(self, nlp_qualifier, text):
        # Arrange
        ca = ContextAlgorithm(nlp=nlp_qualifier)
        doc = nlp_qualifier(text)

        # Act
        doc = ca(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Experiencer.Family" in doc.spans[SPANS_KEY][0]._.qualifiers_str

    def test_experiencer_transformer(self, nlp_qualifier, text):
        # Arrange
        et = ExperiencerTransformer(nlp=nlp_qualifier)
        doc = nlp_qualifier(text)

        # Act
        doc = et(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Experiencer.Family" in doc.spans[SPANS_KEY][0]._.qualifiers_str

    def test_negation_transformer(self, nlp_qualifier, text):
        # Arrange
        et = NegationTransformer(nlp=nlp_qualifier)
        doc = nlp_qualifier(text)

        # Act
        doc = et(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert "Presence.Absent" in doc.spans[SPANS_KEY][0]._.qualifiers_str
