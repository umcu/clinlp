import pytest
from spacy.tokens import Span
from tests.conftest import TEST_DATA_DIR

from clinlp.ie import RuleBasedEntityMatcher, Term


def ent_tuples(ents: list[Span]):
    return [(str(ent), ent.start, ent.end, ent.label_) for ent in ents]


class TestRuleBasedEntityMatcher:
    def test_add_term_str(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_term(concept="delier", term="delier")

        # Assert
        assert len(rbem._phrase_matcher) == 1
        assert len(rbem._matcher) == 0

    def test_add_term_dict(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_term(concept="delier", term={"phrase": "delier", "attr": "NORM"})

        # Assert
        assert len(rbem._phrase_matcher) == 0
        assert len(rbem._matcher) == 1

    def test_add_term_list(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_term(concept="delier", term=[{"TEXT": "delier"}])

        # Assert
        assert len(rbem._phrase_matcher) == 0
        assert len(rbem._matcher) == 1

    def test_add_term_term(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_term(concept="delier", term=Term("delier"))

        # Assert
        assert len(rbem._phrase_matcher) == 1
        assert len(rbem._matcher) == 0

    def test_add_term_non_allowed_type(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act & Assert
        with pytest.raises(TypeError):
            rbem.add_term(concept="delier", term=1)

    def test_add_terms(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_terms(
            concept="delier",
            terms=[
                "delier",
                {"phrase": "delier", "attr": "NORM"},
                [{"TEXT": "delier"}],
                Term("delier", pseudo=True),
            ],
        )

        # Assert
        assert len(rbem._phrase_matcher) == 2
        assert len(rbem._matcher) == 2
        assert len(rbem._concepts) == 4
        assert len(rbem._terms) == 4

    def test_add_terms_from_dict(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_terms_from_dict(
            terms={
                "delier": [
                    {"phrase": "delier", "attr": "NORM"},
                    {"phrase": "delier", "pseudo": True},
                ],
                "anemie": ["anemie"],
            }
        )

        # Assert
        assert len(rbem._phrase_matcher) == 2
        assert len(rbem._matcher) == 1
        assert len(rbem._concepts) == 3
        assert len(rbem._terms) == 3

    def test_add_terms_from_json(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_terms_from_json(TEST_DATA_DIR / "rbem_terms.json")

        # Assert
        assert len(rbem._phrase_matcher) == 2
        assert len(rbem._matcher) == 1
        assert len(rbem._concepts) == 3
        assert len(rbem._terms) == 3

    def test_add_terms_from_csv(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)

        # Act
        rbem.add_terms_from_csv(TEST_DATA_DIR / "rbem_terms.csv")

        # Assert
        assert len(rbem._phrase_matcher) == 4
        assert len(rbem._matcher) == 2
        assert len(rbem._concepts) == 6
        assert len(rbem._terms) == 6

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", []),
            ("dhr was delirantt", []),
        ],
    )
    def test_match_simple(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(concept="delier", terms=["delier", "delirant"])

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", [("Delirant", 2, 3, "delier")]),
            ("dhr was delirantt", []),
        ],
    )
    def test_match_attr(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(
            concept="delier",
            terms=[
                Term("delier", attr="LOWER"),
                Term("delirant", attr="LOWER"),
            ],
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("dhr was kluts kwijt", [("kluts kwijt", 2, 4, "delier")]),
            ("dhr was kluts even kwijt", [("kluts even kwijt", 2, 5, "delier")]),
            ("dhr was kluts gister en vandaag kwijt", []),
        ],
    )
    def test_match_proximity(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(
            concept="delier",
            terms=[Term("delier"), Term("kluts kwijt", proximity=1)],
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("dhr was delirant", [("delirant", 2, 3, "delier")]),
            ("dhr was Delirant", [("Delirant", 2, 3, "delier")]),
            ("dhr was delirantt", [("delirantt", 2, 3, "delier")]),
        ],
    )
    def test_match_fuzzy(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(
            concept="delier", terms=[Term("delier"), Term("delirant", fuzzy=1)]
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
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
        rbem.add_term(
            concept="bloeding", term=Term("bloeding graad ii", fuzzy=1, fuzzy_min_len=6)
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("onrustige indruk", [("onrustige", 0, 1, "delier")]),
            ("onrustige benen", []),
        ],
    )
    def test_match_pseudo(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(
            concept="delier",
            terms=["onrustige", Term("onrustige benen", pseudo=True)],
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("onrustige indruk", [("onrustige", 0, 1, "delier")]),
            ("onrustige benen", [("onrustige", 0, 1, "delier")]),
        ],
    )
    def test_match_pseudo_different_concepts(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_term(concept="delier", term="onrustige")
        rbem.add_term(concept="geen_delier", term=Term("onrustige benen", pseudo=True))

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
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
        rbem.add_terms(
            concept="delier",
            terms=["delier", "delirant", "kluts kwijt", Term("onrustig", fuzzy=0)],
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
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
    def test_rbem_level_term_settings_overwrite(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, attr="LOWER", fuzzy=1)
        rbem.add_terms(
            concept="delier",
            terms=["delier", Term("delirant", fuzzy=0), Term("DOS", attr="TEXT")],
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    @pytest.mark.parametrize(
        ("text", "expected_entities"),
        [
            ("delier", [("delier", 0, 1, "delier")]),
            ("delirant", [("delirant", 0, 1, "delier")]),
            ("delirium", [("delirium", 0, 1, "delier")]),
        ],
    )
    def test_match_mixed_patterns(self, nlp, text, expected_entities):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_terms(
            concept="delier", terms=["delier", Term("delirant"), [{"TEXT": "delirium"}]]
        )

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == expected_entities

    def test_match_mixed_concepts(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp)
        rbem.add_term(concept="bloeding", term="bloeding")
        rbem.add_term(concept="prematuriteit", term="prematuur")
        text = "complicaties door bloeding bij prematuur"

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == [
            ("bloeding", 2, 3, "bloeding"),
            ("prematuur", 4, 5, "prematuriteit"),
        ]

    def test_resolve_overlap(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)
        rbem.add_terms(
            concept="slokdarmatresie", terms=["atresie", "oesophagus atresie"]
        )
        text = "patient heeft oesophagus atresie"

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == [("oesophagus atresie", 2, 4, "slokdarmatresie")]

    def test_resolve_overlap_adjacent(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=True)
        rbem.add_terms(concept="anemie", terms=["erytrocyten", "transfusie"])
        text = "patient kreeg erytrocyten transfusie"

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == [
            ("erytrocyten", 2, 3, "anemie"),
            ("transfusie", 3, 4, "anemie"),
        ]

    def test_resolve_overlap_disabled(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, resolve_overlap=False)
        rbem.add_terms(
            concept="slokdarmatresie", terms=["atresie", "oesophagus atresie"]
        )
        text = "patient heeft oesophagus atresie"

        # Act
        entities = ent_tuples(rbem.match_entities(nlp(text)))

        # Assert
        assert entities == [
            ("oesophagus atresie", 2, 4, "slokdarmatresie"),
            ("atresie", 3, 4, "slokdarmatresie"),
        ]

    def test_spans_key(self, nlp):
        # Arrange
        rbem = RuleBasedEntityMatcher(nlp=nlp, spans_key="custom_key")
        rbem.add_term(concept="delier", term="delier")
        text = "patient heeft delier"

        # Act
        doc = rbem(nlp(text))

        # Assert
        assert doc.spans["custom_key"][0].label_ == "delier"
