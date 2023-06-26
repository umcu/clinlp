import spacy

from clinlp.qualifier.rule_based import parse_rules

def show_demo():

    nlp = spacy.blank("clinlp")

    # Entities
    ruler = nlp.add_pipe('entity_ruler')

    terms = {
        'diagnose': ['epiduraal hematoom', 'heupdysplasie', 'longempyeem', 'giardiasis', 'cardiomyopathie',
                     'voorwandinfarct', 'MI', 'TI', 'pneumonitis', 'osteoperose', 'osteopenie', 'diffusiestoornis',
                     'gaswisselingsstoornis', 'restrictieve longfunctie', 'tumor', 'borstkanker', 'otitis media',
                     'hematurie', 'lymfoom'],
        'symptoom': ['koorts', 'hoesten', 'grieperig', 'vermoeidheid', 'futloos', 'moe', 'vocht vasthouden',
                     'kortademigheid', 'dyspnoe', 'verkouden', 'onrustig', 'paralyse', 'diarree', 'obstructie', 'pijn',
                     'ziek', 'pijnklachten', 'POB', 'palpitaties', 'dyspnoe d\'effort', 'oedeem', 'oedemen', 'ortopnoe',
                     'misselijk', 'verminderde intake', 'afvallen', 'bloeding', 'bloedverlies', 'stolsels', 'blush',
                     'alert', 'wheezing', 'intrekkingen', 'souffle', 'contracties', 'syncope', 'aankleuring']
    }

    for term_description, terms in terms.items():
        ruler.add_patterns([{'label': term_description, 'pattern': term} for term in terms])

    # Context
    qualifier_data = {
        'qualifier_classes': [
            {'qualifier': 'Negation', 'levels': ['AFFIRMED', 'NEGATED']},
            {'qualifier': 'Temporality', 'levels': ["CURRENT", "HISTORICAL"]},
        ],
        "rules": [
            {'pattern': "geen", "qualifier": "Negation.NEGATED", "direction": "preceding"},
            {'pattern': "geen afname", "qualifier": "Negation.NEGATED", "direction": "pseudo"},
            {'pattern': "wel", "qualifier": "Negation.NEGATED", "direction": "termination"},
            {'pattern': "weken geleden", "qualifier": "Temporality.HISTORICAL", "direction": "following"},
        ]
    }

    clinlp_qualifier = nlp.add_pipe('clinlp_qualifier')
    clinlp_qualifier.add_rules(parse_rules(input_json='resources/default_qualifiers.json'))

    text = (
        "Patiente bij mij gezien op spreekuur die pneumonitis vorige maand heeft gehad. Zij had geen last meer van "
        "kortademigheid, wel was er nog sprake van hoesten, geen afname vermoeidheid."
    )

    doc = nlp(text)

    print("Doc: ")
    print(doc)

    print("\nTokens: ")
    for i, token in enumerate(doc):
        print(f"{str(i):2s}", ">" if token.is_sent_start else " ", token)

    print("\nEntities: ")
    for ent in doc.ents:
        print(f"{str((ent.start, ent.end)):10s}", f"{ent.label_:12s}", f"{str(ent):20s}", ent._.qualifiers)


def show_load_rules():

    from clinlp.qualifier.rule_based import parse_rules

    rules = parse_rules("clinlp/qualifier_classes.json")

    for rule in rules:
        print(rule)

def recode_json():

    import json
    from collections import defaultdict

    with open('clinlp/resources/psynlp_context_rules.json', 'rb') as file:
        data = json.load(file)

    patterns = defaultdict(list)

    for pattern in data['rules']:
        patterns[(pattern['qualifier'], pattern['direction'])].append(pattern['pattern'])

    patterns_2 = []

    for key, value in patterns.items():

        patterns_2.append({
            'qualifier': key[0],
            'direction': key[1],
            'patterns': value
        })

    data['rules'] = patterns_2

    from pprint import pprint

    pprint(data)

    with open('resources/default_qualifiers_newformat.json', 'w') as file:
        json.dump(data, file)


def load_resources():

    from importlib import resources
    import json

    with open(resources.path('clinlp.resources', 'psynlp_context_rules.json'), 'rb') as file:
        data = json.load(file)

    print(data)

def check_versioning():

    import spacy

    nlp = spacy.blank('clinlp')

    nlp.to_disk("test_version_model")

    print(nlp.meta)

def check_load_model():

    import spacy

    spacy.load("test_version_model")


if __name__ == '__main__':
    check_load_model()

