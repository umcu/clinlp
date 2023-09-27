import pytest
import spacy

import clinlp
from clinlp.entity import EntityMatcher, Term


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


def ents(doc):
    return [(str(ent), ent.start, ent.end, ent.label_) for ent in doc.ents]


class TestTerm:
    def test_create_term(self):
        assert Term(phrase="diabetes")
        assert Term(phrase="diabetes", attr="NORM").attr == "NORM"
        assert Term(phrase="diabetes", proximity=1).proximity == 1
        assert Term(phrase="diabetes", fuzzy=1).fuzzy == 1
        assert Term(phrase="diabetes", fuzzy_min_len=1).fuzzy_min_len == 1
        assert Term(phrase="diabetes", pseudo=True).pseudo

    def test_spacy_pattern_simple(self, nlp):
        t = Term(phrase="diabetes", attr="NORM")

        assert t.to_spacy_pattern(nlp) == [{"NORM": "diabetes"}]

    def test_spacy_pattern_proximity(self, nlp):
        t = Term("kluts kwijt", proximity=1)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": "kluts"}, {"OP": "?"}, {"TEXT": "kwijt"}]

    def test_spacy_pattern_fuzzy(self, nlp):
        t = Term(phrase="diabetes", fuzzy=3)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": {"FUZZY3": "diabetes"}}]

    def test_spacy_pattern_fuzzy_min_len(self, nlp):
        t = Term(phrase="bloeding graad iv", fuzzy=1, fuzzy_min_len=6)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": {"FUZZY1": "bloeding"}}, {"TEXT": "graad"}, {"TEXT": "iv"}]

    def test_spacy_pattern_pseudo(self, nlp):
        t = Term(phrase="diabetes", pseudo=True)

        assert t.to_spacy_pattern(nlp) == [{"TEXT": "diabetes"}]


class TestClinlpNer:
    def test_create_clinlpner(self, nlp):
        assert EntityMatcher(nlp=nlp)

    def test_use_phrase_matcher(self, nlp):
        assert EntityMatcher(nlp=nlp)._use_phrase_matcher
        assert EntityMatcher(nlp=nlp, attr="NORM")._use_phrase_matcher
        assert not EntityMatcher(nlp=nlp, proximity=1)._use_phrase_matcher
        assert not EntityMatcher(nlp=nlp, fuzzy=1)._use_phrase_matcher
        assert not EntityMatcher(nlp=nlp, fuzzy_min_len=1)._use_phrase_matcher
        assert EntityMatcher(nlp=nlp, pseudo=1)._use_phrase_matcher

    def test_load_concepts(self, nlp):
        ner = EntityMatcher(nlp=nlp)

        concepts = {
            "concept_1": [Term("term1", fuzzy=1), "term2", [{"TEXT": {"FUZZY1": "term3"}}]],
            "concept_2": ["term4", Term("term5"), [{"NORM": "term6"}]],
        }

        ner.load_concepts(concepts)

        assert len(ner._phrase_matcher) + len(ner._matcher) == 6
        assert len(ner._terms) == 6

    def test_load_concepts_nondefault(self, nlp):
        ner = EntityMatcher(nlp=nlp, attr="NORM", fuzzy=1, fuzzy_min_len=10)

        concepts = {
            "concept_1": [Term("term1", fuzzy=1), "term2", [{"TEXT": {"FUZZY1": "term3"}}]],
            "concept_2": ["term4", Term("term5"), [{"NORM": "term6"}]],
        }

        ner.load_concepts(concepts)

        assert len(ner._phrase_matcher) + len(ner._matcher) == 6
        assert len(ner._terms) == 6

    def test_match_overwrite(self, nlp):
        ner = EntityMatcher(nlp=nlp, attr="LOWER", fuzzy=1)

        concepts = {"delier": ["delier", Term("delirant", fuzzy=0), Term("DOS", attr="TEXT")]}

        ner.load_concepts(concepts)

        assert ents(ner(nlp("delier"))) == [("delier", 0, 1, "delier")]
        assert ents(ner(nlp("Delier"))) == [("Delier", 0, 1, "delier")]
        assert ents(ner(nlp("delir"))) == [("delir", 0, 1, "delier")]
        assert ents(ner(nlp("delirant"))) == [("delirant", 0, 1, "delier")]
        assert ents(ner(nlp("delirnt"))) == []
        assert ents(ner(nlp("Delirant"))) == [("Delirant", 0, 1, "delier")]
        assert ents(ner(nlp("dos"))) == []
        assert ents(ner(nlp("doss"))) == []
        assert ents(ner(nlp("DOS"))) == [("DOS", 0, 1, "delier")]

    def test_match(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts({"delier": ["delier", "delirant"]})

        assert ents(ner(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(ner(nlp("dhr was Delirant"))) == []
        assert ents(ner(nlp("dhr was delirantt"))) == []

    def test_match_attr(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts(
            {
                "delier": [
                    Term("delier", attr="LOWER"),
                    Term("delirant", attr="LOWER"),
                ]
            }
        )

        assert ents(ner(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(ner(nlp("dhr was Delirant"))) == [("Delirant", 2, 3, "delier")]
        assert ents(ner(nlp("dhr was delirantt"))) == []

    def test_match_proximity(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts({"delier": [Term("delier"), Term("kluts kwijt", proximity=1)]})

        assert ents(ner(nlp("dhr was kluts kwijt"))) == [("kluts kwijt", 2, 4, "delier")]
        assert ents(ner(nlp("dhr was kluts even kwijt"))) == [("kluts even kwijt", 2, 5, "delier")]
        assert ents(ner(nlp("dhr was kluts gister en vandaag kwijt"))) == []

    def test_match_fuzzy(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts({"delier": [Term("delier"), Term("delirant", fuzzy=1)]})

        assert ents(ner(nlp("dhr was delirant"))) == [("delirant", 2, 3, "delier")]
        assert ents(ner(nlp("dhr was Delirant"))) == [("Delirant", 2, 3, "delier")]
        assert ents(ner(nlp("dhr was delirantt"))) == [("delirantt", 2, 3, "delier")]

    def test_match_fuzzy_min_len(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts(
            {
                "bloeding": [
                    Term("bloeding graad ii", fuzzy=1, fuzzy_min_len=6),
                ]
            }
        )

        assert ents(ner(nlp("bloeding graad ii"))) == [("bloeding graad ii", 0, 3, "bloeding")]
        assert ents(ner(nlp("Bloeding graad ii"))) == [("Bloeding graad ii", 0, 3, "bloeding")]
        assert ents(ner(nlp("bleoding graad ii"))) == []
        assert ents(ner(nlp("bbloeding graad ii"))) == [("bbloeding graad ii", 0, 3, "bloeding")]
        assert ents(ner(nlp("bloeding graadd ii"))) == []

    def test_match_pseudo(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts({"delier": ["onrustige", Term("onrustige benen", pseudo=True)]})

        assert ents(ner(nlp("onrustige indruk"))) == [("onrustige", 0, 1, "delier")]
        assert ents(ner(nlp("onrustige benen"))) == []

    def test_match_pseudo_different_concepts(self, nlp):
        ner = EntityMatcher(nlp=nlp)
        ner.load_concepts({"delier": ["onrustige"], "geen_delier": [Term("onrustige benen", pseudo=True)]})

        assert ents(ner(nlp("onrustige indruk"))) == [("onrustige", 0, 1, "delier")]
        assert ents(ner(nlp("onrustige benen"))) == [("onrustige", 0, 1, "delier")]

    def test_ner_level_term_settings(self, nlp):
        ner = EntityMatcher(nlp=nlp, attr="LOWER", proximity=1, fuzzy=1, fuzzy_min_len=5)

        ner.load_concepts({"delier": ["delier", "delirant", "kluts kwijt", Term("onrustig", fuzzy=0)]})

        assert ents(ner(nlp("was kluts even kwijt"))) == [("kluts even kwijt", 1, 4, "delier")]
        assert ents(ner(nlp("wekt indruk delierant te zijn"))) == [("delierant", 2, 3, "delier")]
        assert ents(ner(nlp("status na Delier"))) == [("Delier", 2, 3, "delier")]
        assert ents(ner(nlp("onrustig"))) == [("onrustig", 0, 1, "delier")]
        assert ents(ner(nlp("onrustigg"))) == []

    def test_match_mixed_patterns(self, nlp):
        ner = EntityMatcher(nlp=nlp)

        ner.load_concepts({"delier": ["delier", Term("delirant"), [{"TEXT": "delirium"}]]})

        assert ents(ner(nlp("delier"))) == [("delier", 0, 1, "delier")]
        assert ents(ner(nlp("delirant"))) == [("delirant", 0, 1, "delier")]
        assert ents(ner(nlp("delirium"))) == [("delirium", 0, 1, "delier")]

    def test_match_mixed_concepts(self, nlp):
        ner = EntityMatcher(nlp=nlp)

        ner.load_concepts(
            {
                "bloeding": [
                    "bloeding",
                ],
                "prematuriteit": ["prematuur"],
            }
        )

        assert ents(ner(nlp("complicaties door bloeding bij prematuur"))) == [
            ("bloeding", 2, 3, "bloeding"),
            ("prematuur", 4, 5, "prematuriteit"),
        ]

    def test_match_overlap(self, nlp):
        ner = EntityMatcher(nlp=nlp)

        ner.load_concepts({"slokdarmatresie": ["atresie", "oesophagus atresie"]})

        assert ents(ner(nlp("patient heeft oesophagus atresie"))) == [("oesophagus atresie", 2, 4, "slokdarmatresie")]
