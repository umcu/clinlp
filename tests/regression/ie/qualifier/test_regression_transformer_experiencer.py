import json

import pytest
import spacy

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture()
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")

    # ruler
    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    # recognizer
    _ = nlp.add_pipe(
        "clinlp_experiencer_transformer",
        config={"token_window": 32, "placeholder": "X"},
    )

    return nlp


class TestRegressionTransformer:
    def test_qualifier_cases(self, nlp):
        with open("tests/data/qualifier_cases.json", "rb") as file:
            data = json.load(file)

        incorrect_ents = set()

        for example in data["examples"]:
            doc = nlp(example["text"])

            predicted_ent = doc.spans[SPANS_KEY][0]
            example_ent = example["ent"]

            try:
                assert predicted_ent.start == example_ent["start"]
                assert predicted_ent.end == example_ent["end"]
                assert str(predicted_ent) == example_ent["text"]
                assert getattr(predicted_ent._, ATTR_QUALIFIERS_STR).issubset(
                    set(example_ent["qualifiers"])
                )

            except AssertionError:

                print(
                    f"Incorrect (#{example['example_id']}): "
                    f"text={example['text']}, "
                    f"example_ent={example_ent}, "
                    f"predicted qualifiers="
                    f"{getattr(predicted_ent._, ATTR_QUALIFIERS_STR)}"
                )

                incorrect_ents.add(example['example_id'])


        assert incorrect_ents == {32, 75, 76}
