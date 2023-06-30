import json

import pytest
import spacy

import clinlp


@pytest.fixture()
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    # ruler
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    # recognizer
    _ = nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

    return nlp


class TestRegressionContextAlgorithm:
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
                    assert predicted_ent._.qualifiers == set(example_ent["qualifiers"])
                except AssertionError:
                    print(
                        f"Incorrect (#{example_ent['ent_id']}): text={example['text']}, example_ent={example_ent}, predicted qualifiers={predicted_ent._.qualifiers}"
                    )
                    incorrect_ents.add(example_ent["ent_id"])

        assert incorrect_ents == set()