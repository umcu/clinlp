import pytest
from tests.conftest import _make_nlp, _make_nlp_entity
from tests.regression import load_qualifier_examples

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR

KNOWN_FAILURES = {
    9,
    16,
    18,
    31,
    32,
    43,
    51,
    52,
    59,
    62,
    63,
    64,
    66,
    67,
    68,
}

examples = load_qualifier_examples("qualifier_cases.json", KNOWN_FAILURES)


@pytest.fixture(scope="class")
def nlp():
    return _make_nlp()


@pytest.fixture(scope="class")
def nlp_entity(nlp):
    return _make_nlp_entity(nlp)


@pytest.fixture(scope="class")
def nlp_qualifier(nlp_entity):
    _ = nlp_entity.add_pipe(
        "clinlp_negation_transformer", config={"token_window": 32, "placeholder": "X"}
    )

    return nlp_entity


class TestRegressionNegationTransformer:
    @pytest.mark.parametrize("text, expected_ent", examples)
    def test_regression_negation_transformer(self, nlp_qualifier, text, expected_ent):
        # Act
        doc = nlp_qualifier(text)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert doc.spans[SPANS_KEY][0].start == expected_ent["start"]
        assert doc.spans[SPANS_KEY][0].end == expected_ent["end"]
        assert str(doc.spans[SPANS_KEY][0]) == expected_ent["text"]
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR).issubset(
            set(expected_ent["qualifiers"])
        )
