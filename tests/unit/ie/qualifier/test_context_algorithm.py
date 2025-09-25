import pytest
from spacy.language import Vocab
from spacy.tokens import Doc, Span
from tests.conftest import TEST_DATA_DIR

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ContextAlgorithm,
    ContextRule,
    ContextRuleDirection,
    QualifierClass,
)
from clinlp.ie.qualifier.context_algorithm import _MatchedContextPattern
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR


# Arrange
@pytest.fixture
def nlp_ca(nlp_entity):
    nlp_entity.add_pipe("clinlp_sentencizer")
    return nlp_entity


# Arrange
@pytest.fixture
def ca(nlp):
    return ContextAlgorithm(nlp=nlp, load_rules=False)


# Arrange
@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["dit", "is", "een", "test"])


# Arrange
@pytest.fixture
def mock_qualifier_class():
    return QualifierClass("Mock", ["Mock_1", "Mock_2"])


class TestUnitContextRule:
    def test_create_context_rule_string_pattern(self):
        # Arrange
        pattern = "test"
        direction = ContextRuleDirection.PRECEDING
        qualifier = QualifierClass("Negation", ["Affirmed", "Negated"]).create(
            "Negated"
        )

        # Act
        qr = ContextRule(pattern, direction, qualifier)

        # Assert
        assert qr.pattern == pattern
        assert qr.direction == direction
        assert qr.qualifier == qualifier

    def test_create_context_rule_spacy_pattern(self):
        # Arrange
        pattern = [{"LOWER": "test"}]
        direction = ContextRuleDirection.PRECEDING
        qualifier = QualifierClass("Negation", ["Affirmed", "Negated"]).create(
            "Negated"
        )

        # Act
        qr = ContextRule(pattern, direction, qualifier)

        # Assert
        assert qr.pattern == pattern
        assert qr.direction == direction
        assert qr.qualifier == qualifier


class TestUnitMatchedQualifierPattern:
    def test_create_mqp_pattern(self, mock_qualifier_class):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.PRECEDING,
            qualifier=mock_qualifier_class.create("Mock_1"),
        )
        start = 0
        end = 10

        # Act
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)

        # Assert
        assert mqp.rule is rule
        assert mqp.start == start
        assert mqp.end == end
        assert mqp.scope is None

    def test_create_mqp_with_offset(self, mock_qualifier_class):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.PRECEDING,
            qualifier=mock_qualifier_class.create("Mock_1"),
        )
        start = 0
        end = 10
        offset = 25

        # Act
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end, offset=offset)

        # Assert
        assert mqp.rule is rule
        assert mqp.start == start + offset
        assert mqp.end == end + offset
        assert mqp.scope is None

    def test_mqp_initialize_scope_preceding(self, mock_qualifier_class, mock_doc):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.PRECEDING,
            qualifier=mock_qualifier_class.create("Mock_1"),
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_mqp_initialize_scope_following(self, mock_qualifier_class, mock_doc):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.FOLLOWING,
            qualifier=mock_qualifier_class.create("Mock_1"),
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (0, 2)

    def test_mqp_initialize_scope_bidirectional(self, mock_qualifier_class, mock_doc):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.BIDIRECTIONAL,
            qualifier=mock_qualifier_class.create("Mock_1"),
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (0, 4)

    def test_mqp_initialize_scope_preceding_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.PRECEDING,
            qualifier=mock_qualifier_class.create("Mock_1"),
            max_scope=1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_mqp_initialize_scope_following_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.FOLLOWING,
            qualifier=mock_qualifier_class.create("Mock_1"),
            max_scope=1,
        )
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_mqp_initialize_scope_bidirectional_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.BIDIRECTIONAL,
            qualifier=mock_qualifier_class.create("Mock_1"),
            max_scope=1,
        )
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.set_initial_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_mqp_initialize_scope_invalid_scope(self, mock_qualifier_class, mock_doc):
        # Arrange
        rule = ContextRule(
            pattern="_",
            direction=ContextRuleDirection.FOLLOWING,
            qualifier=mock_qualifier_class.create("Mock_1"),
            max_scope=-1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Assert
        with pytest.raises(ValueError, match=r".*max_scope must be at least 1.*"):
            # Act
            mqp.set_initial_scope(sentence=sentence)


class TestUnitContextAlgorithm:
    def test_create_context_algorithm(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": [[{"LOWER": "geleden"}]],
                    "qualifier": "Temporality.Historical",
                    "direction": "following",
                },
            ],
        }

        # Act
        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)

        # Assert
        assert len(ca.rules) == len(rules["rules"])
        assert len(ca._matcher) == 1
        assert len(ca._phrase_matcher) == 1

    def test_parse_qualifier(self, mock_qualifier_class, ca):
        # Arrange
        value = "Mock.Mock_1"
        qualifier_classes = {"Mock": mock_qualifier_class}
        expected_qualifier = mock_qualifier_class.create("Mock_1")

        # Act
        parsed_qualifier = ca._parse_qualifier(value, qualifier_classes)

        # Assert
        assert parsed_qualifier == expected_qualifier

    def test_parse_qualifier_error(self, mock_qualifier_class, ca):
        # Arrange
        value = "Mock_Mock_1"
        qualifier_classes = {"Mock": mock_qualifier_class}

        # Assert
        with pytest.raises(ValueError, match=r".*Cannot parse qualifier.*"):
            # Act
            ca._parse_qualifier(value, qualifier_classes)

    def test_load_default_rules(self, nlp_ca):
        # Act
        ca = ContextAlgorithm(nlp=nlp_ca)

        # Assert
        assert len(ca.rules) > 100

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("preceding", ContextRuleDirection.PRECEDING),
            ("following", ContextRuleDirection.FOLLOWING),
            ("bidirectional", ContextRuleDirection.BIDIRECTIONAL),
            ("pseudo", ContextRuleDirection.PSEUDO),
            ("termination", ContextRuleDirection.TERMINATION),
        ],
    )
    def test_parse_direction(self, ca, direction, expected):
        # Act
        parsed_direction = ca._parse_direction(direction)

        # Assert
        assert parsed_direction == expected

    def test_parse_rules_data(self, ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical", "Future"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "max_scope": 5,
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": ["weken geleden"],
                    "qualifier": "Temporality.Historical",
                    "direction": "following",
                },
                {
                    "patterns": ["preventief"],
                    "qualifier": "Temporality.Future",
                    "direction": "bidirectional",
                },
            ],
        }

        # Act
        rules = ca._parse_rules(rules=rules)

        # Assert
        assert len(rules) == 3
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.Negated"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[0].max_scope == 5
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.Historical"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"
        assert rules[1].max_scope is None
        assert rules[2].pattern == "preventief"
        assert str(rules[2].qualifier) == "Temporality.Future"
        assert str(rules[2].direction) == "ContextRuleDirection.BIDIRECTIONAL"
        assert rules[2].max_scope is None

    def test_parse_rules_json(self, ca):
        # Act
        rules = ca._parse_rules(
            rules=str(TEST_DATA_DIR / "qualifier_rules_simple.json")
        )

        # Assert
        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.Negated"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[0].max_scope == 5
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.Historical"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"
        assert rules[1].max_scope is None

    def test_add_rule(self, ca):
        # Arrange
        rule = ContextRule(
            pattern="test",
            qualifier=QualifierClass("Mock", ["Mock_1", "Mock_2"]).create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )

        # Act
        ca.add_rule(rule)

        # Assert
        assert len(ca.rules) == 1
        assert next(iter(ca.rules.values())) == rule

    def test_add_rules(self, ca):
        # Arrange
        rule_1 = ContextRule(
            pattern="test",
            qualifier=QualifierClass("Mock", ["Mock_1", "Mock_2"]).create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )

        rule_2 = ContextRule(
            pattern="test",
            qualifier=QualifierClass("Mock", ["Mock_1", "Mock_2"]).create("Mock_2"),
            direction=ContextRuleDirection.FOLLOWING,
        )

        # Act
        ca.add_rules([rule_1, rule_2])
        rules = iter(ca.rules.values())

        # Assert
        assert len(ca.rules) == 2
        assert next(rules) == rule_1
        assert next(rules) == rule_2

    def test_get_sentences_with_entities(self, nlp_ca, ca):
        # Arrange
        text = "Patient 1 heeft ENTITY. Patient 2 niet. Patient 3 heeft ook ENTITY."
        doc = nlp_ca(text)

        # Act
        sents = ca._get_sentences_with_entities(doc)

        # Assert
        assert len(sents) == 2
        for sent, ents in sents.items():
            assert "ENTITY" in str(sent)
            assert len(ents) == 1

    def test_resolve_matched_pattern_conflicts(self, nlp_ca, ca):
        # Arrange
        doc = nlp_ca("mogelijk ENTITY uitgesloten")
        ent = doc.spans[SPANS_KEY][0]

        qualifier_class = QualifierClass(
            name="Presence",
            values=["Absent", "Uncertain", "Present"],
            default="Present",
        )

        rule1 = ContextRule(
            pattern="uitgesloten",
            qualifier=qualifier_class.create("Absent"),
            direction=ContextRuleDirection.FOLLOWING,
        )
        rule2 = ContextRule(
            pattern="mogelijk",
            qualifier=qualifier_class.create("Uncertain"),
            direction=ContextRuleDirection.PRECEDING,
        )

        pattern1 = _MatchedContextPattern(rule=rule1, start=0, end=1)
        pattern2 = _MatchedContextPattern(rule=rule2, start=2, end=3)

        # Act
        resolved_patterns = ca._resolve_matched_pattern_conflicts(
            ent, [pattern1, pattern2]
        )

        # Assert
        assert resolved_patterns == [pattern2]

    def test_call_no_ents(self, nlp_ca, ca):
        # Arrange
        text = "tekst zonder entities"
        old_doc = nlp_ca(text)

        # Act
        new_doc = ca(old_doc)

        # Assert
        assert new_doc == old_doc

    def test_call_no_rules(self, nlp_ca, ca):
        # Arrange
        text = "Patient heeft ENTITY (wel ents, geen rules)"
        doc = nlp_ca(text)

        # Assert
        with pytest.raises(RuntimeError):
            # Act
            ca(doc)

    def test_call_preceding(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Patient heeft geen ENTITY."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_preceding_with_default(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Patient heeft geen ENTITY."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Current" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_preceding_multiple_ents(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Patient heeft geen ENTITY of ENTITY."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_following_multiple_ents(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["uitgesloten"],
                    "qualifier": "Negation.Negated",
                    "direction": "following",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Aanwezigheid van ENTITY of ENTITY is uitgesloten."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_bidirectional_multiple_ents(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Temporality", "values": ["Historical", "Current"]},
            ],
            "rules": [
                {
                    "patterns": ["als tiener"],
                    "qualifier": "Temporality.Historical",
                    "direction": "bidirectional",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "ENTITY als tiener ENTITY"
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert len(doc.spans[SPANS_KEY]) == 2
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_pseudo(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": ["geen toename"],
                    "qualifier": "Negation.Negated",
                    "direction": "pseudo",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Er is geen toename van ENTITY."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_termination_preceding(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": ["maar"],
                    "qualifier": "Negation.Negated",
                    "direction": "termination",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Er is geen ENTITY, maar wel ENTITY."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_termination_directly_preceding(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Plausibility", "values": ["Plausible", "Hypothetical"]},
            ],
            "rules": [
                {
                    "patterns": ["mogelijk"],
                    "qualifier": "Plausibility.Hypothetical",
                    "direction": "following",
                },
                {
                    "patterns": [","],
                    "qualifier": "Plausibility.Hypothetical",
                    "direction": "termination",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "ENTITY, mogelijk ENTITY"
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_termination_following(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["uitgesloten"],
                    "qualifier": "Negation.Negated",
                    "direction": "following",
                },
                {
                    "patterns": ["maar"],
                    "qualifier": "Negation.Negated",
                    "direction": "termination",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Mogelijk ENTITY, maar ENTITY uitgesloten."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_termination_directly_following(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Plausibility", "values": ["Plausible", "Hypothetical"]},
            ],
            "rules": [
                {
                    "patterns": ["mogelijk"],
                    "qualifier": "Plausibility.Hypothetical",
                    "direction": "preceding",
                },
                {
                    "patterns": ["op basis van"],
                    "qualifier": "Plausibility.Hypothetical",
                    "direction": "termination",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "ENTITY mogelijk op basis van ENTITY"
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_multiple_sentences(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Er is geen ENTITY. Daarnaast ENTITY onderzocht."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_multiple_qualifiers(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": ["als kind"],
                    "qualifier": "Temporality.Historical",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Heeft als kind geen ENTITY gehad."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_terminate_multiple_qualifiers(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
                {
                    "patterns": [","],
                    "qualifier": "Negation.Negated",
                    "direction": "termination",
                },
                {
                    "patterns": ["als kind"],
                    "qualifier": "Temporality.Historical",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Heeft als kind geen ENTITY, wel ENTITY gehad."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_call_multiple_patterns(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen", "subklinische"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "Liet subklinisch ONHERKEND_ENTITY en geen ENTITY zien."
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_overlap_rule_and_ent(self, nlp):
        # Arrange
        nlp.add_pipe("clinlp_sentencizer")
        ruler = nlp.add_pipe("clinlp_rule_based_entity_matcher")
        ruler.add_term(concept="entity", term="geen eetlust")

        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
            ],
            "rules": [
                {
                    "patterns": ["geen"],
                    "qualifier": "Negation.Negated",
                    "direction": "preceding",
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        text = "Patient laat weten geen eetlust te hebben"
        doc = nlp(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_multiple_of_same_qualifier(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {
                    "name": "Presence",
                    "values": ["Absent", "Uncertain", "Present"],
                    "default": "Present",
                },
            ],
            "rules": [
                {
                    "qualifier": "Presence.Absent",
                    "direction": "following",
                    "patterns": ["uitgesloten"],
                },
                {
                    "qualifier": "Presence.Uncertain",
                    "direction": "preceding",
                    "patterns": ["mogelijk"],
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "mogelijk ENTITY is uitgesloten"
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Presence.Absent" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Presence.Uncertain" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_call_multiple_of_same_qualifier_with_priorities(self, nlp_ca):
        # Arrange
        rules = {
            "qualifiers": [
                {
                    "name": "Presence",
                    "values": ["Absent", "Uncertain", "Present"],
                    "default": "Present",
                    "priorities": {"Absent": 2, "Uncertain": 1, "Present": 0},
                },
            ],
            "rules": [
                {
                    "qualifier": "Presence.Uncertain",
                    "direction": "preceding",
                    "patterns": ["mogelijk"],
                },
                {
                    "qualifier": "Presence.Absent",
                    "direction": "following",
                    "patterns": ["uitgesloten"],
                },
            ],
        }

        ca = ContextAlgorithm(nlp=nlp_ca, rules=rules)
        text = "mogelijk ENTITY uitgesloten"
        doc = nlp_ca(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Presence.Absent" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Presence.Uncertain" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
