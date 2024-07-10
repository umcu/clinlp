"""
Converts interesting sentences (to be input) into a format that
is useful for regression tests. It pre-codes the qualifiers based
on the context algorithm as a convenience, but check these manually
before adding them.
"""

import itertools
from pprint import pprint

import spacy
from spacy.language import Language

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY


def get_model() -> Language:
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    # Entities
    terms = {
        "named_entity": ["ENTITY"],
    }

    entity_matcher = nlp.add_pipe(
        "clinlp_rule_based_entity_matcher", config={"attr": "NORM"}
    )
    entity_matcher.add_terms_from_dict(terms)

    # Qualifiers
    nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

    return nlp


if __name__ == "__main__":
    nlp = get_model()

    texts = [
        "CMV/EBV ENTITY negatief, lues negatief",
        "CMV/EBV HIV negatief, ENTITY negatief",
    ]

    data = []
    start_example_id = 77

    cntr = itertools.count()

    for text in texts:
        doc = nlp(text)

        ents = doc.spans[SPANS_KEY]

        if len(ents) != 1:
            msg = f"Expected 1 entity, got {len(ents)}."
            raise ValueError(msg)

        ent = ents[0]

        qualifiers = []

        for qualifier in ent._.qualifiers:
            name, value = qualifier.split(".", 1)
            qualifiers.append({"name": name, "value": value})

        data.append(
            {
                "identifier": start_example_id + next(cntr),
                "text": text,
                "annotations": [
                    {
                        "text": str(ent),
                        "start": ent.start,
                        "end": ent.end,
                        "label": "entity",
                        "qualifiers": qualifiers,
                    }
                ],
            }
        )

    pprint(data, sort_dicts=False)
