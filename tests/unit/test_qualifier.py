import pytest
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from clinlp.component import ContextRule, ContextRuleDirection, Qualifier
from clinlp.component.qualifier import (
    _MatchedContextPattern,
    _parse_direction,
    _parse_qualifier,
    parse_rules,
)


@pytest.fixture
def mock_qualifier():
    return Qualifier("MOCK", ["MOCK_1", "MOCK_2"])


@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["dit", "is", "een", "test"])


class TestUnitQualifier:
    def test_qualifier(self):
        q = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"])

        assert q["AFFIRMED"]
        assert q["NEGATED"]


class TestUnitQualifierRuleDirection:
    def test_qualifier_rule_direction_create(self):
        assert ContextRuleDirection.PRECEDING
        assert ContextRuleDirection.FOLLOWING
        assert ContextRuleDirection.PSEUDO
        assert ContextRuleDirection.TERMINATION


class TestUnitQualifierRule:
    def test_create_qualifier_rule_1(self):
        pattern = "test"
        level = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"]).NEGATED
        direction = ContextRuleDirection.PRECEDING

        qr = ContextRule(pattern, level, direction)

        assert qr.pattern == pattern
        assert qr.qualifier == level
        assert qr.direction == direction

    def test_create_qualifier_rule_2(self):
        pattern = [{"LOWER": "test"}]
        level = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"]).NEGATED
        direction = ContextRuleDirection.PRECEDING

        qr = ContextRule(pattern, level, direction)

        assert qr.pattern == pattern
        assert qr.qualifier == level
        assert qr.direction == direction


class TestUnitMatchedQualifierPattern:
    def test_create_matched_qualifier_pattern(self, mock_qualifier):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.PRECEDING)
        start = 0
        end = 10

        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)

        assert mqp.rule is rule
        assert mqp.start == start
        assert mqp.end == end
        assert mqp.scope is None

    def test_create_matched_qualifier_pattern_with_offset(self, mock_qualifier):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.PRECEDING)
        start = 0
        end = 10
        offset = 25

        mqp = _MatchedContextPattern(rule=rule, start=start, end=end, offset=offset)

        assert mqp.rule is rule
        assert mqp.start == start + offset
        assert mqp.end == end + offset
        assert mqp.scope is None

    def test_matched_qualifier_pattern_initial_scope_preceding(self, mock_qualifier, mock_doc):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.PRECEDING)
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_matched_qualifier_pattern_initial_scope_following(self, mock_qualifier, mock_doc):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.FOLLOWING)
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (0, 2)

    def test_matched_qualifier_pattern_initial_scope_preceding_with_max_scope(self, mock_qualifier, mock_doc):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.PRECEDING)
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence, max_scope=1)

        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_following_with_max_scope(self, mock_qualifier, mock_doc):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.FOLLOWING)
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.set_initial_scope(sentence=sentence, max_scope=1)

        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_invalid_scope(self, mock_qualifier, mock_doc):
        rule = ContextRule(pattern="_", qualifier=mock_qualifier.MOCK_1, direction=ContextRuleDirection.FOLLOWING)
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        with pytest.raises(ValueError):
            mqp.set_initial_scope(sentence=sentence, max_scope=0)


class TestUnitLoadRules:
    def test_parse_level(self, mock_qualifier):
        level = "MOCK.MOCK_1"
        qualifiers = {"MOCK": mock_qualifier}

        assert _parse_qualifier(level, qualifiers) == mock_qualifier.MOCK_1

    def test_parse_level_unhappy(self, mock_qualifier):
        level = "MOCK_MOCK_1"
        qualifiers = {"MOCK": mock_qualifier}

        with pytest.raises(ValueError):
            _parse_qualifier(level, qualifiers)

    def test_parse_direction(self):
        assert _parse_direction("preceding") == ContextRuleDirection.PRECEDING
        assert _parse_direction("following") == ContextRuleDirection.FOLLOWING
        assert _parse_direction("pseudo") == ContextRuleDirection.PSEUDO
        assert _parse_direction("termination") == ContextRuleDirection.TERMINATION

    def test_load_rules_data(self):
        data = {
            "qualifiers": [
                {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
            ],
            "rules": [
                {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                {"patterns": ["weken geleden"], "qualifier": "Temporality.HISTORICAL", "direction": "following"},
            ],
        }

        rules = parse_rules(data=data)

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.NEGATED"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.HISTORICAL"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"

    def test_load_rules_json(self):
        rules = parse_rules(input_json="tests/data/qualifier_rules_simple.json")

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.NEGATED"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.HISTORICAL"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"

    def test_load_rules_unhappy(self):
        with pytest.raises(ValueError):
            parse_rules(input_json="_.json", data={"a": "b"})
