import pytest

import clinlp  # noqa: F401
from clinlp.ie import SPANS_KEY, RuleBasedEntityMatcher, Term, create_concept_dict


def ents(doc):
    return [(str(ent), ent.start, ent.end, ent.label_) for ent in doc.spans[SPANS_KEY]]


class TestCreateConceptDict:
    def test_create_concept_dict(self):
        # Act
        concepts = create_concept_dict("tests/data/concept_examples.csv")

        # Assert
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
    @pytest.mark.parametrize(
        "rbem_kwargs, expected_use_phrase_matcher",
        [
            ({}, True),
            ({"attr": "NORM"}, True),
            ({"proximity": 1}, False),
            ({"fuzzy": 1}, False),
            ({"fuzzy_min_len": 1}, False),
            ({"pseudo": 1}, True),
        ],
    )
    def test_use_phrase_matcher(self, nlp, rbem_kwargs, expected_use_phrase_matcher):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, **rbem_kwargs)

        # Act
        use_phrase_matcher = rbem._use_phrase_matcher

        # Assert
        assert use_phrase_matcher == expected_use_phrase_matcher

    def test_load_concepts(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
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

        # Assert
        assert len(rbem._phrase_matcher) + len(rbem._matcher) == 6
        assert len(rbem._terms) == 6

    def test_load_concepts_nondefault(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, attr="NORM", fuzzy=1, fuzzy_min_len=10)

        # Act
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

        # Assert
        assert len(rbem._phrase_matcher) + len(rbem._matcher) == 6
        assert len(rbem._terms) == 6

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", []),
            ("dhr was delirantt", []),
        ],
    )
    def test_match_simple(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts({"delier": ["delier", "delirant"]})

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", [("Delirant", 2, 3, "delier")]),
            ("dhr was delirantt", []),
        ],
    )
    def test_match_attr(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "delier": [
                    Term("delier", attr="LOWER"),
                    Term("delirant", attr="LOWER"),
                ]
            }
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("dhr was kluts kwijt", [("kluts kwijt", 2, 4, "delier")]),
            ("dhr was kluts even kwijt", [("kluts even kwijt", 2, 5, "delier")]),
            ("dhr was kluts gister en vandaag kwijt", []),
        ],
    )
    def test_match_proximity(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {"delier": [Term("delier"), Term("kluts kwijt", proximity=1)]}
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", [("Delirant", 2, 3, "delier")]),
            ("dhr was delirantt", [("delirantt", 2, 3, "delier")]),
        ],
    )
    def test_match_fuzzy(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts({"delier": [Term("delier"), Term("delirant", fuzzy=1)]})

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("bloeding graad ii", [("bloeding graad ii", 0, 3, "bloeding")]),
            ("Bloeding graad ii", [("Bloeding graad ii", 0, 3, "bloeding")]),
            ("bleoding graad ii", []),
            ("bbloeding graad ii", [("bbloeding graad ii", 0, 3, "bloeding")]),
            ("bloeding graadd ii", []),
        ],
    )
    def test_match_fuzzy_min_len(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "bloeding": [
                    Term("bloeding graad ii", fuzzy=1, fuzzy_min_len=6),
                ]
            }
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("onrustige indruk", [("onrustige", 0, 1, "delier")]),
            ("onrustige benen", []),
        ],
    )
    def test_match_pseudo(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {"delier": ["onrustige", Term("onrustige benen", pseudo=True)]}
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("onrustige indruk", [("onrustige", 0, 1, "delier")]),
            ("onrustige benen", [("onrustige", 0, 1, "delier")]),
        ],
    )
    def test_match_pseudo_different_concepts(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "delier": ["onrustige"],
                "geen_delier": [Term("onrustige benen", pseudo=True)],
            }
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("was kluts even kwijt", [("kluts even kwijt", 1, 4, "delier")]),
            ("wekt indruk delierant te zijn", [("delierant", 2, 3, "delier")]),
            ("status na Delier", [("Delier", 2, 3, "delier")]),
            ("onrustig", [("onrustig", 0, 1, "delier")]),
            ("onrustigg", []),
        ],
    )
    def test_rbem_level_term_settings(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(
            nlp=nlp, attr="LOWER", proximity=1, fuzzy=1, fuzzy_min_len=5
        )
        rbem.load_concepts(
            {"delier": ["delier", "delirant", "kluts kwijt", Term("onrustig", fuzzy=0)]}
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("delier", [("delier", 0, 1, "delier")]),
            ("Delier", [("Delier", 0, 1, "delier")]),
            ("delir", [("delir", 0, 1, "delier")]),
            ("delirant", [("delirant", 0, 1, "delier")]),
            ("delirnt", []),
            ("Delirant", [("Delirant", 0, 1, "delier")]),
            ("dos", []),
            ("doss", []),
            ("DOS", [("DOS", 0, 1, "delier")]),
        ],
    )
    def test_match_overwrite_rbem_level_settings(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, attr="LOWER", fuzzy=1)
        rbem.load_concepts(
            {"delier": ["delier", Term("delirant", fuzzy=0), Term("DOS", attr="TEXT")]}
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        "text, expected_entities",
        [
            ("delier", [("delier", 0, 1, "delier")]),
            ("delirant", [("delirant", 0, 1, "delier")]),
            ("delirium", [("delirium", 0, 1, "delier")]),
        ],
    )
    def test_match_mixed_patterns(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {"delier": ["delier", Term("delirant"), [{"TEXT": "delirium"}]]}
        )

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == expected_entities

    def test_match_mixed_concepts(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.load_concepts(
            {
                "bloeding": [
                    "bloeding",
                ],
                "prematuriteit": ["prematuur"],
            }
        )
        text = "complicaties door bloeding bij prematuur"

        # Act
        entities = ents(rbem(nlp(text)))

        # Assert
        assert entities == [
            ("bloeding", 2, 3, "bloeding"),
            ("prematuur", 4, 5, "prematuriteit"),
        ]

    def test_resolve_overlap(self, nlp):
        # Arrange
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)
        ner.load_concepts({"slokdarmatresie": ["atresie", "oesophagus atresie"]})
        text = "patient heeft oesophagus atresie"

        # Act
        entities = ents(ner(nlp(text)))

        # Assert
        assert entities == [("oesophagus atresie", 2, 4, "slokdarmatresie")]

    def test_resolve_overlap_adjacent(self, nlp):
        # Arrange
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)
        ner.load_concepts({"anemie": ["erytrocyten", "transfusie"]})
        text = "patient kreeg erytrocyten transfusie"

        # Act
        entities = ents(ner(nlp(text)))

        # Assert
        assert entities == [
            ("erytrocyten", 2, 3, "anemie"),
            ("transfusie", 3, 4, "anemie"),
        ]

    def test_no_resolve_overlap(self, nlp):
        # Arrange
        ner = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=False)
        ner.load_concepts({"slokdarmatresie": ["atresie", "oesophagus atresie"]})
        text = "patient heeft oesophagus atresie"

        # Act
        entities = ents(ner(nlp(text)))

        # Assert
        assert entities == [
            ("oesophagus atresie", 2, 4, "slokdarmatresie"),
            ("atresie", 3, 4, "slokdarmatresie"),
        ]
