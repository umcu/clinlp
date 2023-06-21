import pytest
import spacy

from clinlp.component.qualifier import ContextMatcher, parse_rules


@pytest.fixture
def nlp():
    nlp = spacy.blank("clinlp")
    nlp.add_pipe("clinlp_sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])

    return nlp


class TestUnitQualifierMatcher:
    def test_create_qualifier_matcher(self, nlp):
        data = {
            "qualifiers": [
                {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
            ],
            "rules": [
                {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                {"patterns": [[{"LOWER": "geleden"}]], "qualifier": "Temporality.HISTORICAL", "direction": "following"},
            ],
        }

        rules = parse_rules(data=data)

        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)

        assert len(qm.rules) == len(rules)
        assert len(qm._matcher) == 1
        assert len(qm._phrase_matcher) == 1

    def test_get_sentences_having_entity(self, nlp):
        text = "Patient 1 heeft SYMPTOOM. Patient 2 niet. Patient 3 heeft ook SYMPTOOM."
        doc = nlp(text)

        sents = list(ContextMatcher._get_sentences_having_entity(doc))

        assert len(sents) == 2
        for sent in sents:
            assert "SYMPTOOM" in str(sent)

    def test_match_qualifiers_no_ents(self, nlp):
        text = "tekst zonder entities"
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None)
        old_doc = nlp(text)

        new_doc = qm(old_doc)

        assert new_doc == old_doc

    def test_match_qualifiers_no_rules(self, nlp):
        text = "Patient heeft SYMPTOOM (wel ents, geen rules)"
        doc = nlp(text)
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None)

        with pytest.raises(RuntimeError):
            qm(doc)

    def test_match_qualifiers_preceding(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )
        text = "Patient heeft geen SYMPTOOM."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers

    def test_match_qualifiers_preceding_multiple_ents(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )
        text = "Patient heeft geen SYMPTOOM of SYMPTOOM."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_following_multiple_ents(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["uitgesloten"], "qualifier": "Negation.NEGATED", "direction": "following"},
                ],
            }
        )
        text = "Aanwezigheid van SYMPTOOM of SYMPTOOM is uitgesloten."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_pseudo(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                    {"patterns": ["geen toename"], "qualifier": "Negation.NEGATED", "direction": "pseudo"},
                ],
            }
        )
        text = "Er is geen toename van SYMPTOOM."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" not in doc.ents[0]._.qualifiers

    def test_match_qualifiers_termination_preceding(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                    {"patterns": ["maar"], "qualifier": "Negation.NEGATED", "direction": "termination"},
                ],
            }
        )
        text = "Er is geen SYMPTOOM, maar wel SYMPTOOM."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers

    def test_match_qualifiers_termination_following(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["uitgesloten"], "qualifier": "Negation.NEGATED", "direction": "following"},
                    {"patterns": ["maar"], "qualifier": "Negation.NEGATED", "direction": "termination"},
                ],
            }
        )
        text = "Mogelijk SYMPTOOM, maar SYMPTOOM uitgesloten."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" not in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_multiple_sentences(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )

        text = "Er is geen SYMPTOOM. Daarnaast SYMPTOOM onderzocht."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers

    def test_match_qualifier_multiple_qualifiers(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                    {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                    {"patterns": ["als kind"], "qualifier": "Temporality.HISTORICAL", "direction": "preceding"},
                ],
            }
        )

        text = "Heeft als kind geen SYMPTOOM gehad."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[0]._.qualifiers

    def test_match_qualifier_terminate_multiple_qualifiers(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                    {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                    {"patterns": [","], "qualifier": "Negation.NEGATED", "direction": "termination"},
                    {"patterns": ["als kind"], "qualifier": "Temporality.HISTORICAL", "direction": "preceding"},
                ],
            }
        )

        text = "Heeft als kind geen SYMPTOOM, wel SYMPTOOM gehad."
        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[0]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[1]._.qualifiers

    def test_match_qualifier_multiple_patterns(self, nlp):
        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen", "subklinische"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )

        text = "Liet subklinisch ONHERKEND_SYMPTOOM en geen SYMPTOOM zien."

        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers

    def test_overlap_rule_and_ent(self):
        nlp = spacy.blank("clinlp")
        nlp.add_pipe("clinlp_sentencizer")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([{"label": "symptoom", "pattern": "geen eetlust"}])

        rules = parse_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"patterns": ["geen"], "qualifier": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )

        text = "Patient laat weten geen eetlust te hebben"

        qm = ContextMatcher(nlp=nlp, name="_", default_rules=None, rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" not in doc.ents[0]._.qualifiers

    def test_load_default_rules(self, nlp):
        qm = ContextMatcher(nlp=nlp, name="_")

        assert len(qm.rules) > 100
