"""
Converts interesting sentences (to be input) into a format that
is useful for regression tests. It pre-codes the qualifiers based
on the context algorithm as a convenience, but check these manually
before adding them.
"""

import itertools
from pprint import pprint

import spacy

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY


def get_model():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    # Entities
    concepts = {
        "named_entity": ["ENTITY"],
    }

    entity_matcher = nlp.add_pipe(
        "clinlp_rule_based_entity_matcher", config={"attr": "NORM"}
    )
    entity_matcher.load_concepts(concepts)

    # Qualifiers
    nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

    return nlp


if __name__ == "__main__":
    nlp = get_model()

    texts = [
        "CMV/EBV ENTITY negatief, ENTITY negatief",
    ]

    data = []
    ent_id = 57

    cntr = itertools.count()

    for text in texts:
        doc = nlp(text)

        data.append(
            {
                "text": text,
                "ents": [
                    {
                        "ent_id": ent_id + next(cntr),
                        "start": ent.start,
                        "end": ent.end,
                        "text": str(ent),
                        "qualifiers": list(ent._.qualifiers_str),
                    }
                    for ent in doc.spans[SPANS_KEY]
                ],
            }
        )

    pprint(data, sort_dicts=False)
