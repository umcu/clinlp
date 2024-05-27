import json

import pytest
import spacy

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR

KNOWN_FAILURES = {9, 11, 12, 32}


@pytest.fixture(scope="module")
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    _ = nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

    return nlp


with open("tests/data/qualifier_cases.json", "rb") as file:
    data = json.load(file)

examples = []

for example in data["examples"]:
    mark = pytest.mark.xfail if example["example_id"] in KNOWN_FAILURES else []

    examples.append(
        pytest.param(example["text"], example["ent"], id="qualifier_case_", marks=mark)
    )


class TestRegressionContextAlgorithm:
    @pytest.mark.parametrize("text, expected_ent", examples)
    def test_qualifier_cases(self, nlp, text, expected_ent):
        # Act
        doc = nlp(text)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 1
        assert doc.spans[SPANS_KEY][0].start == expected_ent["start"]
        assert doc.spans[SPANS_KEY][0].end == expected_ent["end"]
        assert str(doc.spans[SPANS_KEY][0]) == expected_ent["text"]
        assert getattr(doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR).issubset(
            set(expected_ent["qualifiers"])
        )
