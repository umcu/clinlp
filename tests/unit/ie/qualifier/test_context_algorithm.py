import pytest
import spacy
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from clinlp.ie import SPANS_KEY
from clinlp.ie.qualifier import (
    ContextAlgorithm,
    ContextRule,
    ContextRuleDirection,
    QualifierClass,
)
from clinlp.ie.qualifier.context_algorithm import _MatchedContextPattern
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture
def nlp_entity():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")
    ruler = nlp.add_pipe("span_ruler", config={"spans_key": SPANS_KEY})
    ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])

    return nlp


@pytest.fixture
def mock_qualifier_class():
    return QualifierClass("Mock", ["Mock_1", "Mock_2"])


@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["dit", "is", "een", "test"])


@pytest.fixture
def ca(nlp):
    return ContextAlgorithm(nlp=nlp, load_rules=False)


class TestUnitQualifierRule:
    def test_create_qualifier_rule_1(self):
        # Arrange
        pattern = "test"
        qualifier = QualifierClass("Negation", ["Affirmed", "Negated"]).create(
            "Negated"
        )
        direction = ContextRuleDirection.PRECEDING

        # Act
        qr = ContextRule(pattern, qualifier, direction)

        # Assert
        assert qr.pattern == pattern
        assert qr.qualifier == qualifier
        assert qr.direction == direction

    def test_create_qualifier_rule_2(self):
        # Arrange
        pattern = [{"LOWER": "test"}]
        qualifier = QualifierClass("Negation", ["Affirmed", "Negated"]).create(
            "Negated"
        )
        direction = ContextRuleDirection.PRECEDING

        # Act
        qr = ContextRule(pattern, qualifier, direction)

        # Assert
        assert qr.pattern == pattern
        assert qr.qualifier == qualifier
        assert qr.direction == direction


class TestUnitMatchedQualifierPattern:
    def test_create_matched_qualifier_pattern(self, mock_qualifier_class):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
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

    def test_create_matched_qualifier_pattern_with_offset(self, mock_qualifier_class):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
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

    def test_matched_qualifier_pattern_initial_scope_preceding(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_matched_qualifier_pattern_initial_scope_following(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (0, 2)

    def test_matched_qualifier_pattern_initial_scope_bidirectional(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.BIDIRECTIONAL,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (0, 4)

    def test_matched_qualifier_pattern_initial_scope_preceding_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
            max_scope=1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_following_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
            max_scope=1,
        )
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_bidirectional_with_max_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.BIDIRECTIONAL,
            max_scope=1,
        )
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Act
        mqp.initialize_scope(sentence=sentence)

        # Assert
        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_matched_qualifier_pattern_initial_scope_invalid_scope(
        self, mock_qualifier_class, mock_doc
    ):
        # Arrange
        rule = ContextRule(
            pattern="_",
            qualifier=mock_qualifier_class.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
            max_scope=-1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        # Assert
        with pytest.raises(ValueError):
            # Act
            mqp.initialize_scope(sentence=sentence)


class TestUnitContextAlgorithm:
    def test_create_qualifier_matcher(self, nlp_entity):
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
        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)

        # Assert
        assert len(ca.rules) == len(rules["rules"])
        assert len(ca._matcher) == 1
        assert len(ca._phrase_matcher) == 1

    def test_parse_value(self, mock_qualifier_class, ca):
        # Arrange
        value = "Mock.Mock_1"
        qualifier_factories = {"Mock": mock_qualifier_class}
        expected_qualifier = mock_qualifier_class.create("Mock_1")

        # Act
        parsed_qualifier = ca._parse_qualifier(value, qualifier_factories)

        # Assert
        assert parsed_qualifier == expected_qualifier

    def test_parse_value_error(self, mock_qualifier_class, ca):
        # Arrange
        value = "Mock_Mock_1"
        qualifiers = {"Mock": mock_qualifier_class}

        # Assert
        with pytest.raises(ValueError):
            # Act
            ca._parse_qualifier(value, qualifiers)

    @pytest.mark.parametrize(
        "direction, expected",
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

    def test_load_rules_data(self, ca):
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

    def test_load_rules_json(self, ca):
        # Act
        rules = ca._parse_rules(rules="tests/data/qualifier_rules_simple.json")

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

    def test_get_sentences_with_entities(self, nlp_entity, ca):
        # Arrange
        text = "Patient 1 heeft SYMPTOOM. Patient 2 niet. Patient 3 heeft ook SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        sents = ca._get_sentences_with_entities(doc)

        # Assert
        assert len(sents) == 2
        for sent, ents in sents.items():
            assert "SYMPTOOM" in str(sent)
            assert len(ents) == 1

    def test_resolve_matched_pattern_conflicts(self, nlp_entity, ca):
        # Arrange
        doc = nlp_entity("mogelijk SYMPTOOM uitgesloten")
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

    def test_match_qualifiers_no_ents(self, nlp_entity, ca):
        # Arrange
        text = "tekst zonder entities"
        old_doc = nlp_entity(text)

        # Act
        new_doc = ca(old_doc)

        # Assert
        assert new_doc == old_doc

    def test_match_qualifiers_no_rules(self, nlp_entity, ca):
        # Arrange
        text = "Patient heeft SYMPTOOM (wel ents, geen rules)"
        doc = nlp_entity(text)

        # Assert
        with pytest.raises(RuntimeError):
            # Act
            ca(doc)

    def test_match_qualifiers_preceding(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Patient heeft geen SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_preceding_with_default(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Patient heeft geen SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Current" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_preceding_multiple_ents(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Patient heeft geen SYMPTOOM of SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_following_multiple_ents(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Aanwezigheid van SYMPTOOM of SYMPTOOM is uitgesloten."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_bidirectional_multiple_ents(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "SYMPTOOM als tiener SYMPTOOM"
        doc = nlp_entity(text)

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

    def test_match_qualifiers_pseudo(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Er is geen toename van SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_termination_preceding(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Er is geen SYMPTOOM, maar wel SYMPTOOM."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_termination_directly_preceding(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "SYMPTOOM, mogelijk SYMPTOOM"
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_termination_following(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Mogelijk SYMPTOOM, maar SYMPTOOM uitgesloten."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_termination_directly_following(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "SYMPTOOM mogelijk op basis van SYMPTOOM"
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Plausibility.Hypothetical" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifiers_multiple_sentences(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Er is geen SYMPTOOM. Daarnaast SYMPTOOM onderzocht."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Negation.Negated" not in getattr(
            doc.spans[SPANS_KEY][1]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifier_multiple_qualifiers(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Heeft als kind geen SYMPTOOM gehad."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Temporality.Historical" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_match_qualifier_terminate_multiple_qualifiers(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Heeft als kind geen SYMPTOOM, wel SYMPTOOM gehad."
        doc = nlp_entity(text)

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

    def test_match_qualifier_multiple_patterns(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "Liet subklinisch ONHERKEND_SYMPTOOM en geen SYMPTOOM zien."
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Negation.Negated" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_overlap_rule_and_ent(self, nlp):
        # Arrange
        nlp.add_pipe("clinlp_sentencizer")
        ruler = nlp.add_pipe("clinlp_rule_based_entity_matcher")
        ruler.load_concepts({"symptoom": ["geen eetlust"]})

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

    def test_multiple_matches_of_same_qualifier(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "mogelijk SYMPTOOM is uitgesloten"
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Presence.Absent" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Presence.Uncertain" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_multiple_matches_of_same_qualifier_with_priorities(self, nlp_entity):
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

        ca = ContextAlgorithm(nlp=nlp_entity, rules=rules)
        text = "mogelijk SYMPTOOM uitgesloten"
        doc = nlp_entity(text)

        # Act
        doc = ca(doc)

        # Assert
        assert "Presence.Absent" in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )
        assert "Presence.Uncertain" not in getattr(
            doc.spans[SPANS_KEY][0]._, ATTR_QUALIFIERS_STR
        )

    def test_load_default_rules(self, nlp_entity):
        # Act
        ca = ContextAlgorithm(nlp=nlp_entity)

        # Assert
        assert len(ca.rules) > 100
