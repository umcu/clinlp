import pytest
from tests.conftest import _make_nlp, _make_nlp_entity
from tests.regression import load_qualifier_examples

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS

KNOWN_FAILURES = {"9", "11", "12", "32"}

examples = load_qualifier_examples("data/qualifier_cases.json", KNOWN_FAILURES)


# Arrange
@pytest.fixture(scope="class")
def nlp():
    return _make_nlp()


# Arrange
@pytest.fixture(scope="class")
def nlp_entity(nlp):
    return _make_nlp_entity(nlp)


# Arrange
@pytest.fixture(scope="class")
def nlp_qualifier(nlp_entity):
    nlp_entity.add_pipe("clinlp_sentencizer")

    nlp_entity.add_pipe(
        "clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"}
    )

    return nlp_entity


class TestRegressionContextAlgorithm:
    @pytest.mark.parametrize(("text", "expected_ent"), examples)
    def test_regression_context_algorithm(self, nlp_qualifier, text, expected_ent):
        # Act
        doc = nlp_qualifier(text)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert str(doc.spans[SPANS_KEY][0]) == expected_ent.text
        assert doc.spans[SPANS_KEY][0].start == expected_ent.start
        assert doc.spans[SPANS_KEY][0].end == expected_ent.end
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS).issubset(
            set(expected_ent.qualifiers)
        )
