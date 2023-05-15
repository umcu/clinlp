import pytest

from clinlp import create_model
from clinlp.qualifier import QualifierMatcher, load_rules

nlp = create_model()
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns([{"label": "symptoom", "pattern": "SYMPTOOM"}])


class TestQualifierMatcher:
    def test_create_qualifier_matcher(self):
        data = {
            "qualifiers": [
                {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
            ],
            "rules": [
                {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                {"pattern": [{"LOWER": "geleden"}], "level": "Temporality.HISTORICAL", "direction": "following"},
            ],
        }

        rules = load_rules(data=data)

        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)

        assert len(qm.rules) == len(rules)
        assert len(qm._matcher) == 1
        assert len(qm._phrase_matcher) == 1

    def test_get_sentences_having_entity(self):
        text = "Patient 1 heeft SYMPTOOM. Patient 2 niet. Patient 3 heeft ook SYMPTOOM."
        doc = nlp(text)

        sents = list(QualifierMatcher._get_sentences_having_entity(doc))

        assert len(sents) == 2
        for sent in sents:
            assert "SYMPTOOM" in str(sent)

    def test_match_qualifiers_no_ents(self):
        text = "tekst zonder entities"
        qm = QualifierMatcher(nlp=nlp, name="_")
        old_doc = nlp(text)

        new_doc = qm(old_doc)

        assert new_doc == old_doc

    def test_match_qualifiers_no_rules(self):
        text = "Patient heeft SYMPTOOM (wel ents, geen rules)"
        doc = nlp(text)
        qm = QualifierMatcher(nlp=nlp, name="_")

        with pytest.raises(RuntimeError):
            qm(doc)

    def test_match_qualifiers_preceding(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )
        text = "Patient heeft geen SYMPTOOM."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers

    def test_match_qualifiers_preceding_multiple_ents(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )
        text = "Patient heeft geen SYMPTOOM of SYMPTOOM."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_following_multiple_ents(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "uitgesloten", "level": "Negation.NEGATED", "direction": "following"},
                ],
            }
        )
        text = "Aanwezigheid van SYMPTOOM of SYMPTOOM is uitgesloten."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_pseudo(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                    {"pattern": "geen toename", "level": "Negation.NEGATED", "direction": "pseudo"},
                ],
            }
        )
        text = "Er is geen toename van SYMPTOOM."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" not in doc.ents[0]._.qualifiers

    def test_match_qualifiers_termination_preceding(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                    {"pattern": "maar", "level": "Negation.NEGATED", "direction": "termination"},
                ],
            }
        )
        text = "Er is geen SYMPTOOM, maar wel SYMPTOOM."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers

    def test_match_qualifiers_termination_following(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "uitgesloten", "level": "Negation.NEGATED", "direction": "following"},
                    {"pattern": "maar", "level": "Negation.NEGATED", "direction": "termination"},
                ],
            }
        )
        text = "Mogelijk SYMPTOOM, maar SYMPTOOM uitgesloten."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" not in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" in doc.ents[1]._.qualifiers

    def test_match_qualifiers_multiple_sentences(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                ],
            }
        )

        text = "Er is geen SYMPTOOM. Daarnaast SYMPTOOM onderzocht."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers

    def test_match_qualifier_multiple_levels(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                    {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                    {"pattern": "als kind", "level": "Temporality.HISTORICAL", "direction": "preceding"},
                ],
            }
        )

        text = "Heeft als kind geen SYMPTOOM gehad."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[0]._.qualifiers

    def test_match_qualifier_terminate_multiple_levels(self):
        rules = load_rules(
            data={
                "qualifiers": [
                    {"qualifier": "Negation", "levels": ["AFFIRMED", "NEGATED"]},
                    {"qualifier": "Temporality", "levels": ["CURRENT", "HISTORICAL"]},
                ],
                "rules": [
                    {"pattern": "geen", "level": "Negation.NEGATED", "direction": "preceding"},
                    {"pattern": ",", "level": "Negation.NEGATED", "direction": "termination"},
                    {"pattern": "als kind", "level": "Temporality.HISTORICAL", "direction": "preceding"},
                ],
            }
        )

        text = "Heeft als kind geen SYMPTOOM, wel SYMPTOOM gehad."
        qm = QualifierMatcher(nlp=nlp, name="_", rules=rules)
        doc = qm(nlp(text))

        assert "Negation.NEGATED" in doc.ents[0]._.qualifiers
        assert "Negation.NEGATED" not in doc.ents[1]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[0]._.qualifiers
        assert "Temporality.HISTORICAL" in doc.ents[1]._.qualifiers
