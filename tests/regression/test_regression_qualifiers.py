import json

from clinlp import create_model
from clinlp.qualifier import load_rules
import pytest

@pytest.fixture()
def nlp():
    nlp = create_model()

    # ruler
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns([{'label': 'named_entity', 'pattern': 'ENTITY'}])

    # recognizer
    qm = nlp.add_pipe('clinlp_qualifier')
    rules = load_rules('resources/default_qualifiers.json')
    qm.add_rules(rules)

    return nlp


class TestRegressionQualifiers:

    def test_qualifier_cases(self, nlp):

        with open('tests/data/qualifier_cases.json', 'rb') as file:
            data = json.load(file)

        for example in data['examples']:

            doc = nlp(example['text'])

            assert len(example['ents']) == len(doc.ents)

            for predicted_ent, example_ent in zip(doc.ents, example['ents']):
                assert predicted_ent.start == example_ent['start']
                assert predicted_ent.end == example_ent['end']
                assert str(predicted_ent) == example_ent['text']
                assert predicted_ent._.qualifiers == set(example_ent['qualifiers'])








