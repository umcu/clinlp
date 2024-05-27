import json

import pytest
from conftest import _make_nlp, _make_nlp_entity

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR

KNOWN_FAILURES = {32, 75, 76}


@pytest.fixture(scope="class")
def nlp():
    return _make_nlp()


@pytest.fixture(scope="class")
def nlp_entity(nlp):
    return _make_nlp_entity(nlp)


@pytest.fixture(scope="class")
def nlp_qualifier(nlp_entity):
    _ = nlp_entity.add_pipe(
        "clinlp_experiencer_transformer",
        config={"token_window": 32, "placeholder": "X"},
    )

    return nlp_entity


with open("tests/data/qualifier_cases.json", "rb") as file:
    data = json.load(file)

examples = []

for example in data["examples"]:
    mark = pytest.mark.xfail if example["example_id"] in KNOWN_FAILURES else []

    examples.append(
        pytest.param(example["text"], example["ent"], id="qualifier_case_", marks=mark)
    )


class TestRegressionExperiencerTransformer:
    @pytest.mark.parametrize("text, expected_ent", examples)
    def test_qualifier_cases(self, nlp_qualifier, text, expected_ent):
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
