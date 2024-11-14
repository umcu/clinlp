"""Model for the Information Extraction demo."""

import itertools

import spacy
from spacy.language import Language

import clinlp  # noqa: F401
from clinlp_apps.ie_demo.src.utils import RESOURCE_PATH

SAMPLE_TERMS_FILE = RESOURCE_PATH / "sample_terms.json"


def get_model() -> Language:
    """
    Create the clinlp model.

    Returns
    -------
        The clinlp model.
    """
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_normalizer")
    nlp.add_pipe("clinlp_sentencizer")

    # Entities
    entity_matcher = nlp.add_pipe(
        "clinlp_rule_based_entity_matcher",
        config={"attr": "NORM", "fuzzy": 1, "fuzzy_min_len": 5},
    )

    entity_matcher.add_terms_from_json(SAMPLE_TERMS_FILE)

    # Qualifiers
    nlp.add_pipe(
        "clinlp_context_algorithm",
        config={"phrase_matcher_attr": "NORM"},
    )

    return nlp


def check_overlapping_entities(doc: spacy.language.Doc) -> bool:
    """
    Check if a document contains overlapping entities.

    Parameters
    ----------
    doc
        The document to check.

    Returns
    -------
        True if the document contains overlapping entities, False otherwise.
    """
    ents = sorted(doc.spans["ents"], key=lambda ent: ent.start)

    return any(ent2.start <= ent1.end for ent1, ent2 in itertools.pairwise(ents))
