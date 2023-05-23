import json

import pytest
import spacy

from clinlp.component.qualifier import load_rules


@pytest.fixture()
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")

    # ruler
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "named_entity", "pattern": "ENTITY"}])

    # recognizer
    qm = nlp.add_pipe("clinlp_qualifier_matcher")
    rules = load_rules("resources/default_qualifiers.json")
    qm.add_rules(rules)

    return nlp


class TestRegressionQualifiers:
    def test_qualifier_cases(self, nlp):
        with open("tests/data/qualifier_cases.json", "rb") as file:
            data = json.load(file)

        for example in data["examples"]:
            doc = nlp(example["text"])

            assert len(example["ents"]) == len(doc.ents)

            for predicted_ent, example_ent in zip(doc.ents, example["ents"]):
                assert predicted_ent.start == example_ent["start"]
                assert predicted_ent.end == example_ent["end"]
                assert str(predicted_ent) == example_ent["text"]
                assert predicted_ent._.qualifiers == set(example_ent["qualifiers"])
