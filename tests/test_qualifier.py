import pytest
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from clinlp.qualifier import (
    MatchedQualifierPattern,
    Qualifier,
    QualifierRule,
    QualifierRuleDirection,
    _parse_direction,
    _parse_level,
    load_rules,
)

mock_qualifier = Qualifier("MOCK", ["MOCK_1", "MOCK_2"])
mock_doc = Doc(Vocab(), words=["dit", "is", "een", "test"])


class TestQualifier:
    def test_qualifier(self):
        q = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"])

        assert q["AFFIRMED"]
        assert q["NEGATED"]


class TestQualifierRuleDirection:
    def test_qualifier_rule_direction_create(self):
        assert QualifierRuleDirection.PRECEDING
        assert QualifierRuleDirection.FOLLOWING
        assert QualifierRuleDirection.PSEUDO
        assert QualifierRuleDirection.TERMINATION


class TestQualifierRule:
    def test_create_qualifier_rule_1(self):
        pattern = "test"
        level = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"]).NEGATED
        direction = QualifierRuleDirection.PRECEDING

        qr = QualifierRule(pattern, level, direction)

        assert qr.pattern == pattern
        assert qr.level == level
        assert qr.direction == direction

    def test_create_qualifier_rule_2(self):
        pattern = [{"LOWER": "test"}]
        level = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"]).NEGATED
        direction = QualifierRuleDirection.PRECEDING

        qr = QualifierRule(pattern, level, direction)

        assert qr.pattern == pattern
        assert qr.level == level
        assert qr.direction == direction


class TestMatchedQualifierPattern:
    def test_create_matched_qualifier_pattern(self):
        rule = QualifierRule(pattern="_", level=mock_qualifier.MOCK_1, direction=QualifierRuleDirection.PRECEDING)
        start = 0
        end = 10

        mqp = MatchedQualifierPattern(rule=rule, start=start, end=end)

        assert mqp.rule is rule
        assert mqp.start == start
        assert mqp.end == end
        assert mqp.scope is None

    def test_create_matched_qualifier_pattern_with_offset(self):
        rule = QualifierRule(pattern="_", level=mock_qualifier.MOCK_1, direction=QualifierRuleDirection.PRECEDING)
        start = 0
        end = 10
        offset = 25

        mqp = MatchedQualifierPattern(rule=rule, start=start, end=end, offset=offset)

        assert mqp.rule is rule
        assert mqp.start == start + offset
        assert mqp.end == end + offset
        assert mqp.scope is None

    def test_matched_qualifier_pattern_initial_scope_preceding(self):
        rule = QualifierRule(pattern="_", level=mock_qualifier.MOCK_1, direction=QualifierRuleDirection.PRECEDING)
        start = 1
        end = 2
        mqp = MatchedQualifierPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_matched_qualifier_pattern_initial_scope_following(self):
        rule = QualifierRule(pattern="_", level=mock_qualifier.MOCK_1, direction=QualifierRuleDirection.FOLLOWING)
        start = 1
        end = 2
        mqp = MatchedQualifierPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (0, 2)


class TestLoadRules:
    def test_parse_level(self):
        level = "MOCK.MOCK_1"
        qualifiers = {"MOCK": mock_qualifier}

        assert _parse_level(level, qualifiers) == mock_qualifier.MOCK_1

    def test_parse_level_unhappy(self):
        level = "MOCK_MOCK_1"
        qualifiers = {"MOCK": mock_qualifier}

        with pytest.raises(ValueError):
            _parse_level(level, qualifiers)

    def test_parse_direction(self):
        assert _parse_direction("preceding") == QualifierRuleDirection.PRECEDING
        assert _parse_direction("following") == QualifierRuleDirection.FOLLOWING
        assert _parse_direction("pseudo") == QualifierRuleDirection.PSEUDO
        assert _parse_direction("termination") == QualifierRuleDirection.TERMINATION

    def test_load_rules_data(self):
        data = {
            "qualifiers": [
                {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
            ],
            "rules": [
                {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                {"pattern": "weken geleden", "level": "Temporality.HISTORICAL", "direction": "following"},
            ],
        }

        rules = load_rules(data=data)

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].level) == "Negation.NEGATED"
        assert str(rules[0].direction) == "QualifierRuleDirection.PRECEDING"
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].level) == "Temporality.HISTORICAL"
        assert str(rules[1].direction) == "QualifierRuleDirection.FOLLOWING"

    def test_load_rules_json(self):
        rules = load_rules(input_json="tests/data/qualifier_rules_simple.json")

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].level) == "Negation.NEGATED"
        assert str(rules[0].direction) == "QualifierRuleDirection.PRECEDING"
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].level) == "Temporality.HISTORICAL"
        assert str(rules[1].direction) == "QualifierRuleDirection.FOLLOWING"

    def test_load_rules_unhappy(self):
        with pytest.raises(ValueError):
            load_rules(input_json="_.json", data={"a": "b"})
