import pytest
from tests.conftest import _make_nlp, _make_nlp_entity
from tests.regression import load_qualifier_examples

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS

KNOWN_FAILURES = {
    "experiencer": {"32", "75", "76"},
    "negation": {
        "9",
        "16",
        "18",
        "31",
        "32",
        "43",
        "51",
        "52",
        "59",
        "62",
        "63",
        "64",
        "66",
        "67",
        "68",
    },
}

examples = {
    tr: load_qualifier_examples("data/qualifier_cases.json", KNOWN_FAILURES[tr])
    for tr in KNOWN_FAILURES
}


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
def nlp_qualifier_negation(nlp_entity):
    _ = nlp_entity.add_pipe(
        "clinlp_negation_transformer", config={"token_window": 32, "placeholder": "X"}
    )

    return nlp_entity


# Arrange
@pytest.fixture(scope="class")
def nlp_qualifier_experiencer(nlp_entity):
    _ = nlp_entity.add_pipe(
        "clinlp_experiencer_transformer",
        config={"token_window": 32, "placeholder": "X"},
    )

    return nlp_entity


class TestRegressionNegationTransformer:
    @pytest.mark.parametrize(("text", "expected_ent"), examples["negation"])
    def test_regression_negation_transformer(
        self, nlp_qualifier_negation, text, expected_ent
    ):
        # Act
        doc = nlp_qualifier_negation(text)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert str(doc.spans[SPANS_KEY][0]) == expected_ent.text
        assert doc.spans[SPANS_KEY][0].start == expected_ent.start
        assert doc.spans[SPANS_KEY][0].end == expected_ent.end
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS).issubset(
            set(expected_ent.qualifiers)
        )


class TestRegressionExperiencerTransformer:
    @pytest.mark.parametrize(("text", "expected_ent"), examples["experiencer"])
    def test_regression_experiencer_transformer(
        self, nlp_qualifier_experiencer, text, expected_ent
    ):
        # Act
        doc = nlp_qualifier_experiencer(text)

        # Assert
        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert str(doc.spans[SPANS_KEY][0]) == expected_ent.text
        assert doc.spans[SPANS_KEY][0].start == expected_ent.start
        assert doc.spans[SPANS_KEY][0].end == expected_ent.end
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS).issubset(
            set(expected_ent.qualifiers)
        )
