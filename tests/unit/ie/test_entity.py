import pytest
import spacy

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY, RuleBasedEntityMatcher, Term, create_concept_dict


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


def ents(doc):
    return [(str(ent), ent.start, ent.end, ent.label_) for ent in doc.spans[SPANS_KEY]]


class TestCreateConceptDict:
    def test_create_concept_dict(self):
        concepts = create_concept_dict("tests/data/concept_examples.csv")

        assert concepts == {
            "hypotensie": [
                Term("hypotensie"),
                Term(
                    phrase="bd verlaagd",
                    proximity=1,
                ),
            ],
            "prematuriteit": [
                Term("prematuriteit"),
                Term(
                    phrase="<p3",
                    attr="TEXT",
                    proximity=1,
                    fuzzy=1,
                    fuzzy_min_len=2,
                    pseudo=False,
                ),
            ],
            "veneus_infarct": [
                Term("veneus infarct"),
                Term(
                    phrase="VI",
                    attr="TEXT",
                ),
            ],
        }


class TestClinlpNer:
    def test_create_clinlpner(self, nlp):
        assert RuleBasedEntityMatcher(nlp=nlp)

    def test_use_phrase_matcher(self, nlp):
        assert RuleBasedEntityMatcher(nlp=nlp)._use_phrase_matcher
        assert RuleBasedEntityMatcher(nlp=nlp, attr="NORM")._use_phrase_matcher
        assert not RuleBasedEntityMatcher(nlp=nlp, proximity=1)._use_phrase_matcher
        assert not RuleBasedEntityMatcher(nlp=nlp, fuzzy=1)._use_phrase_matcher
        assert not RuleBasedEntityMatcher(nlp=nlp, fuzzy_min_len=1)._use_phrase_matcher
        assert RuleBasedEntityMatcher(nlp=nlp, pseudo=1)._use_phrase_matcher

    def test_load_concepts(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        rbem.load_concepts(
            {
                "concept_1": [
                    Term("term1", fuzzy=1),
                    "term2",
                    [{"TEXT": {"FUZZY1": "term3"}}],
                ],
                "concept_2": ["term4", Term("term5"), [{"NORM": "term6"}]],
            }
        )

        assert len(rbem._phrase_matcher) + len(rbem._matcher) == 6
        assert len(rbem._terms) == 6

    def test_load_concepts_nondefault(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp, attr="NORM", fuzzy=1, fuzzy_min_len=10)

        rbem.load_concepts(
            {
                "concept_1": [
                    Term("term1", fuzzy=1),
                    "term2",
                    [{"TEXT": {"FUZZY1": "term3"}}],
                ],
                "concept_2": ["term4", Term("term5"), [{"NORM": "term6"}]],
            }
        )

        assert len(rbem._phrase_matcher) + len(rbem._matcher) == 6
        assert len(rbem._terms) == 6

    def test_match_simpe(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts({"delier": ["delier", "delirant"]})

        assert ents(rbem(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(rbem(nlp("dhr was Delirant"))) == []
        assert ents(rbem(nlp("dhr was delirantt"))) == []

    def test_match_attr(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "delier": [
                    Term("delier", attr="LOWER"),
                    Term("delirant", attr="LOWER"),
                ]
            }
        )

        assert ents(rbem(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(rbem(nlp("dhr was Delirant"))) == [("Delirant", 2, 3, "delier")]
        assert ents(rbem(nlp("dhr was delirantt"))) == []

    def test_match_proximity(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {"delier": [Term("delier"), Term("kluts kwijt", proximity=1)]}
        )

        assert ents(rbem(nlp("dhr was kluts kwijt"))) == [
            ("kluts kwijt", 2, 4, "delier")
        ]
        assert ents(rbem(nlp("dhr was kluts even kwijt"))) == [
            ("kluts even kwijt", 2, 5, "delier")
        ]
        assert ents(rbem(nlp("dhr was kluts gister en vandaag kwijt"))) == []

    def test_match_fuzzy(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts({"delier": [Term("delier"), Term("delirant", fuzzy=1)]})

        assert ents(rbem(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(rbem(nlp("dhr was Delirant"))) == [("Delirant", 2, 3, "delier")]
        assert ents(rbem(nlp("dhr was delirantt"))) == [("delirantt", 2, 3, "delier")]

    def test_match_fuzzy_min_len(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "bloeding": [
                    Term("bloeding graad ii", fuzzy=1, fuzzy_min_len=6),
                ]
            }
        )

        assert ents(rbem(nlp("bloeding graad ii"))) == [
            ("bloeding graad ii", 0, 3, "bloeding")
        ]
        assert ents(rbem(nlp("Bloeding graad ii"))) == [
            ("Bloeding graad ii", 0, 3, "bloeding")
        ]
        assert ents(rbem(nlp("bleoding graad ii"))) == []
        assert ents(rbem(nlp("bbloeding graad ii"))) == [
            ("bbloeding graad ii", 0, 3, "bloeding")
        ]
        assert ents(rbem(nlp("bloeding graadd ii"))) == []

    def test_match_pseudo(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {"delier": ["onrustige", Term("onrustige benen", pseudo=True)]}
        )

        assert ents(rbem(nlp("onrustige indruk"))) == [("onrustige", 0, 1, "delier")]
        assert ents(rbem(nlp("onrustige benen"))) == []

    def test_match_pseudo_different_concepts(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "delier": ["onrustige"],
                "geen_delier": [Term("onrustige benen", pseudo=True)],
            }
        )

        assert ents(rbem(nlp("onrustige indruk"))) == [("onrustige", 0, 1, "delier")]
        assert ents(rbem(nlp("onrustige benen"))) == [("onrustige", 0, 1, "delier")]

    def test_rbem_level_term_settings(self, nlp):
        rbem = RuleBasedEntityMatcher(
            nlp=nlp, attr="LOWER", proximity=1, fuzzy=1, fuzzy_min_len=5
        )

        rbem.load_concepts(
            {"delier": ["delier", "delirant", "kluts kwijt", Term("onrustig", fuzzy=0)]}
        )

        assert ents(rbem(nlp("was kluts even kwijt"))) == [
            ("kluts even kwijt", 1, 4, "delier")
        ]
        assert ents(rbem(nlp("wekt indruk delierant te zijn"))) == [
            ("delierant", 2, 3, "delier")
        ]
        assert ents(rbem(nlp("status na Delier"))) == [("Delier", 2, 3, "delier")]
        assert ents(rbem(nlp("onrustig"))) == [("onrustig", 0, 1, "delier")]
        assert ents(rbem(nlp("onrustigg"))) == []

    def test_match_overwrite_rbem_level_settings(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp, attr="LOWER", fuzzy=1)

        concepts = {
            "delier": ["delier", Term("delirant", fuzzy=0), Term("DOS", attr="TEXT")]
        }

        rbem.load_concepts(concepts)

        assert ents(rbem(nlp("delier"))) == [("delier", 0, 1, "delier")]
        assert ents(rbem(nlp("Delier"))) == [("Delier", 0, 1, "delier")]
        assert ents(rbem(nlp("delir"))) == [("delir", 0, 1, "delier")]
        assert ents(rbem(nlp("delirant"))) == [("delirant", 0, 1, "delier")]
        assert ents(rbem(nlp("delirnt"))) == []
        assert ents(rbem(nlp("Delirant"))) == [("Delirant", 0, 1, "delier")]
        assert ents(rbem(nlp("dos"))) == []
        assert ents(rbem(nlp("doss"))) == []
        assert ents(rbem(nlp("DOS"))) == [("DOS", 0, 1, "delier")]

    def test_match_mixed_patterns(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        rbem.load_concepts(
            {"delier": ["delier", Term("delirant"), [{"TEXT": "delirium"}]]}
        )

        assert ents(rbem(nlp("delier"))) == [("delier", 0, 1, "delier")]
        assert ents(rbem(nlp("delirant"))) == [("delirant", 0, 1, "delier")]
        assert ents(rbem(nlp("delirium"))) == [("delirium", 0, 1, "delier")]

    def test_match_mixed_concepts(self, nlp):
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        rbem.load_concepts(
            {
                "bloeding": [
                    "bloeding",
                ],
                "prematuriteit": ["prematuur"],
            }
        )

        assert ents(rbem(nlp("complicaties door bloeding bij prematuur"))) == [
            ("bloeding", 2, 3, "bloeding"),
            ("prematuur", 4, 5, "prematuriteit"),
        ]

    def test_resolve_overlap(self, nlp):
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)

        ner.load_concepts({"slokdarmatresie": ["atresie", "oesophagus atresie"]})

        assert ents(ner(nlp("patient heeft oesophagus atresie"))) == [
            ("oesophagus atresie", 2, 4, "slokdarmatresie")
        ]

    def test_resolve_overlap_adjacent(self, nlp):
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)

        ner.load_concepts({"anemie": ["erytrocyten", "transfusie"]})

        assert ents(ner(nlp("patient kreeg erytrocyten transfusie"))) == [
            ("erytrocyten", 2, 3, "anemie"),
            ("transfusie", 3, 4, "anemie"),
        ]

    def test_no_resolve_overlap(self, nlp):
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=False)

        ner.load_concepts({"slokdarmatresie": ["atresie", "oesophagus atresie"]})

        assert ents(ner(nlp("patient heeft oesophagus atresie"))) == [
            ("oesophagus atresie", 2, 4, "slokdarmatresie"),
            ("atresie", 3, 4, "slokdarmatresie"),
        ]
