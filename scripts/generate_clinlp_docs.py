"""
Generate a set of clinlp docs as pickle. Useful to regenerate
test data when breaking changes in clinlp.
"""

import pickle
from pathlib import Path

import spacy
from spacy.language import Language

from clinlp.ie import Term

texts = [
    "patient had geen anemie",
    "er was sprake van prematuriteit (<p3), ",
    "een oesophagusatresie was niet uit te sluiten",
    "patient had een prematuur adempatroon",
    "in de voorgeschiedenis geen bloeding",
    "bloeding",
    "na fototherapie verminderde hyperbillirubinaemie",
    "patient aangemeld voor ROP screening",
    "controle op ROP na 4 weken",
    "patient had slechte start",
    "behandeling volgens MIST protocol",
    "familiair belast met anemie",
    "mogelijk dysmatuur als gevolg van TTTS",
    "betreft een Premature zuigeling",
]

terms = {
    "C0002871_anemie": [
        "anemie",
    ],
    "C0020433_hyperbilirubinemie": [
        "hyperbilirubinaemie",
        "fototherapie",
    ],
    "C0015934_intrauterine_groeivertraging": [
        "dysmatuur",
        "<p3",
    ],
    "C0270191_intraventriculaire_bloeding": [
        "bloeding",
    ],
    "C0014850_oesophagus_atresie": [
        "oesophagusatresie",
    ],
    "C0151526_prematuriteit": [
        "prematuriteit",
        "prematuur",
    ],
    "C0035344_retinopathie_van_de_prematuriteit": [
        Term("ROP", attr="TEXT", fuzzy=0),
    ],
}


def get_model() -> Language:
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    entity_matcher = nlp.add_pipe(
        "clinlp_rule_based_entity_matcher",
        config={"attr": "NORM", "fuzzy": 1, "fuzzy_min_len": 8},
    )

    entity_matcher.add_terms_from_dict(terms)

    nlp.add_pipe(
        "clinlp_context_algorithm",
        config={
            "rules": "scripts/mock_context_rules.json",
            "phrase_matcher_attr": "NORM",
        },
    )

    return nlp


if __name__ == "__main__":
    nlp = get_model()

    docs = list(nlp.pipe(texts))

    with Path("clinlp_docs.pickle").open(mode="wb") as f:
        pickle.dump(docs, f)
