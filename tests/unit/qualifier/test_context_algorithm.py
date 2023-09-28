from unittest.mock import patch

import pytest
import spacy
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from clinlp.qualifier import (
    ContextRule,
    ContextRuleDirection,
    Qualifier,
    QualifierFactory,
)
from clinlp.qualifier.context_algorithm import ContextAlgorithm, _MatchedContextPattern
from clinlp.qualifier.qualifier import ATTR_QUALIFIERS_STR


@pytest.fixture
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])

    return nlp


@pytest.fixture
def mock_factory():
    return QualifierFactory("Mock", ["Mock_1", "Mock_2"])


@pytest.fixture
def mock_doc():
    return Doc(Vocab(), words=["dit", "is", "een", "test"])


@pytest.fixture
def ca():
    return ContextAlgorithm(nlp=spacy.blank("clinlp"), load_rules=False)


class TestUnitQualifierRuleDirection:
    def test_qualifier_rule_direction_create(self):
        assert ContextRuleDirection.PRECEDING
        assert ContextRuleDirection.FOLLOWING
        assert ContextRuleDirection.PSEUDO
        assert ContextRuleDirection.TERMINATION


class TestUnitQualifierRule:
    def test_create_qualifier_rule_1(self):
        pattern = "test"
        qualifier = QualifierFactory("Negation", ["Affirmed", "Negated"]).create("Negated")
        direction = ContextRuleDirection.PRECEDING

        qr = ContextRule(pattern, qualifier, direction)

        assert qr.pattern == pattern
        assert qr.qualifier == qualifier
        assert qr.direction == direction

    def test_create_qualifier_rule_2(self):
        pattern = [{"LOWER": "test"}]
        qualifier = QualifierFactory("Negation", ["Affirmed", "Negated"]).create("Negated")
        direction = ContextRuleDirection.PRECEDING

        qr = ContextRule(pattern, qualifier, direction)

        assert qr.pattern == pattern
        assert qr.qualifier == qualifier
        assert qr.direction == direction


class TestUnitMatchedQualifierPattern:
    def test_create_matched_qualifier_pattern(self, mock_factory):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )
        start = 0
        end = 10

        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)

        assert mqp.rule is rule
        assert mqp.start == start
        assert mqp.end == end
        assert mqp.scope is None

    def test_create_matched_qualifier_pattern_with_offset(self, mock_factory):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )
        start = 0
        end = 10
        offset = 25

        mqp = _MatchedContextPattern(rule=rule, start=start, end=end, offset=offset)

        assert mqp.rule is rule
        assert mqp.start == start + offset
        assert mqp.end == end + offset
        assert mqp.scope is None

    def test_matched_qualifier_pattern_initial_scope_preceding(self, mock_factory, mock_doc):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.initialize_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (1, 4)

    def test_matched_qualifier_pattern_initial_scope_following(self, mock_factory, mock_doc):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.initialize_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (0, 2)

    def test_matched_qualifier_pattern_initial_scope_preceding_with_max_scope(self, mock_factory, mock_doc):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.PRECEDING,
            max_scope=1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.initialize_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_following_with_max_scope(self, mock_factory, mock_doc):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
            max_scope=1,
        )
        start = 2
        end = 3
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        mqp.initialize_scope(sentence=sentence)

        assert mqp.scope is not None
        assert mqp.scope == (1, 3)

    def test_matched_qualifier_pattern_initial_scope_invalid_scope(self, mock_factory, mock_doc):
        rule = ContextRule(
            pattern="_",
            qualifier=mock_factory.create("Mock_1"),
            direction=ContextRuleDirection.FOLLOWING,
            max_scope=-1,
        )
        start = 1
        end = 2
        mqp = _MatchedContextPattern(rule=rule, start=start, end=end)
        sentence = Span(mock_doc, start=0, end=4)

        with pytest.raises(ValueError):
            mqp.initialize_scope(sentence=sentence)


class TestUnitContextAlgorithm:
    def test_create_qualifier_matcher(self, nlp):
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

        ca = ContextAlgorithm(nlp=nlp, rules=rules)

        assert len(ca.rules) == len(rules["rules"])
        assert len(ca._matcher) == 1
        assert len(ca._phrase_matcher) == 1

    def test_parse_value(self, mock_factory, ca):
        value = "Mock.Mock_1"
        qualifier_factories = {"Mock": mock_factory}

        assert ca._parse_qualifier(value, qualifier_factories) == mock_factory.create("Mock_1")

    def test_parse_value_unhappy(self, mock_factory, ca):
        value = "Mock_Mock_1"
        qualifiers = {"Mock": mock_factory}

        with pytest.raises(ValueError):
            ca._parse_qualifier(value, qualifiers)

    def test_parse_direction(self, ca):
        assert ca._parse_direction("preceding") == ContextRuleDirection.PRECEDING
        assert ca._parse_direction("following") == ContextRuleDirection.FOLLOWING
        assert ca._parse_direction("pseudo") == ContextRuleDirection.PSEUDO
        assert ca._parse_direction("termination") == ContextRuleDirection.TERMINATION

    def test_load_rules_data(self, ca):
        rules = {
            "qualifiers": [
                {"name": "Negation", "values": ["Affirmed", "Negated"]},
                {"name": "Temporality", "values": ["Current", "Historical"]},
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
            ],
        }

        rules = ca._parse_rules(rules=rules)

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.Negated"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[0].max_scope == 5
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.Historical"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"
        assert rules[1].max_scope is None

    def test_load_rules_json(self, ca):
        rules = ca._parse_rules(rules="tests/data/qualifier_rules_simple.json")

        assert len(rules) == 2
        assert rules[0].pattern == "geen"
        assert str(rules[0].qualifier) == "Negation.Negated"
        assert str(rules[0].direction) == "ContextRuleDirection.PRECEDING"
        assert rules[0].max_scope == 5
        assert rules[1].pattern == "weken geleden"
        assert str(rules[1].qualifier) == "Temporality.Historical"
        assert str(rules[1].direction) == "ContextRuleDirection.FOLLOWING"
        assert rules[1].max_scope is None

    def test_get_sentences_having_entity(self, nlp, ca):
        text = "Patient 1 heeft SYMPTOOM. Patient 2 niet. Patient 3 heeft ook SYMPTOOM."
        doc = nlp(text)

        sents = list(ca._get_sentences_having_entity(doc))

        assert len(sents) == 2
        for sent in sents:
            assert "SYMPTOOM" in str(sent)

    def test_match_qualifiers_no_ents(self, nlp, ca):
        text = "tekst zonder entities"
        old_doc = nlp(text)

        new_doc = ca(old_doc)

        assert new_doc == old_doc

    def test_match_qualifiers_no_rules(self, nlp, ca):
        text = "Patient heeft SYMPTOOM (wel ents, geen rules)"
        doc = nlp(text)

        with pytest.raises(RuntimeError):
            ca(doc)

    def test_match_qualifiers_preceding(self, nlp):
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

        text = "Patient heeft geen SYMPTOOM."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_preceding_with_default(self, nlp):
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

        text = "Patient heeft geen SYMPTOOM."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Temporality.Current" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_preceding_multiple_ents(self, nlp):
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

        text = "Patient heeft geen SYMPTOOM of SYMPTOOM."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_following_multiple_ents(self, nlp):
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

        text = "Aanwezigheid van SYMPTOOM of SYMPTOOM is uitgesloten."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_pseudo(self, nlp):
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

        text = "Er is geen toename van SYMPTOOM."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" not in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_termination_preceding(self, nlp):
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

        text = "Er is geen SYMPTOOM, maar wel SYMPTOOM."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" not in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_termination_following(self, nlp):
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

        text = "Mogelijk SYMPTOOM, maar SYMPTOOM uitgesloten."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" not in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifiers_multiple_sentences(self, nlp):
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

        text = "Er is geen SYMPTOOM. Daarnaast SYMPTOOM onderzocht."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" not in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifier_multiple_qualifiers(self, nlp):
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

        text = "Heeft als kind geen SYMPTOOM gehad."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Temporality.Historical" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifier_terminate_multiple_qualifiers(self, nlp):
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

        text = "Heeft als kind geen SYMPTOOM, wel SYMPTOOM gehad."
        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Negation.Negated" not in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)
        assert "Temporality.Historical" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)
        assert "Temporality.Historical" in getattr(doc.ents[1]._, ATTR_QUALIFIERS_STR)

    def test_match_qualifier_multiple_patterns(self, nlp):
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

        text = "Liet subklinisch ONHERKEND_SYMPTOOM en geen SYMPTOOM zien."

        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_overlap_rule_and_ent(self):
        nlp = spacy.blank("clinlp")
        nlp.add_pipe("clinlp_sentencizer")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([{"label": "symptoom", "pattern": "geen eetlust"}])

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

        text = "Patient laat weten geen eetlust te hebben"

        ca = ContextAlgorithm(nlp=nlp, rules=rules)
        doc = ca(nlp(text))

        assert "Negation.Negated" not in getattr(doc.ents[0]._, ATTR_QUALIFIERS_STR)

    def test_load_default_rules(self, nlp):
        ca = ContextAlgorithm(nlp=nlp)

        assert len(ca.rules) > 100
