import json

import pytest
import spacy

import clinlp  # noqa: F401
from clinlp.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture()
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")

    # ruler
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    # recognizer
    _ = nlp.add_pipe(
        "clinlp_experiencer_transformer", config={"token_window": 32, "placeholder": "X"}
    )

    return nlp

class TestRegressionTransformer:
    def test_qualifier_cases(self, nlp):
        with open("tests/data/qualifier_cases.json", "rb") as file:
            data = json.load(file)

        incorrect_ents = set()

        for example in data["examples"]:
            doc = nlp(example["text"])

            assert len(example["ents"]) == len(doc.ents)

            for predicted_ent, example_ent in zip(doc.ents, example["ents"]):
                try:
                    assert predicted_ent.start == example_ent["start"]
                    assert predicted_ent.end == example_ent["end"]
                    assert str(predicted_ent) == example_ent["text"]
                    assert getattr(predicted_ent._, ATTR_QUALIFIERS_STR).issubset(
                        example_ent["qualifiers"]
                    )
                except AssertionError:
                    print(
                        f"Incorrect (#{example_ent['ent_id']}): "
                        f"text="
                        f"{example['text']}, example_ent={example_ent}, "
                        f"predicted qualifiers="
                        f"{getattr(predicted_ent._, ATTR_QUALIFIERS_STR)}"
                    )
                    incorrect_ents.add(example_ent["ent_id"])

        assert incorrect_ents == {32}