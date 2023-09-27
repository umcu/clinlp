from unittest.mock import patch

import pytest
import spacy
from spacy.tokens import Span

import clinlp
from clinlp.qualifier import Qualifier, QualifierDetector, QualifierFactory
from clinlp.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    ATTR_QUALIFIERS_DICT,
    ATTR_QUALIFIERS_STR,
)


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


@pytest.fixture
def entity():
    doc = spacy.blank("clinlp")("dit is een test")
    return doc[2:3]


@pytest.fixture
def mock_factory():
    return QualifierFactory("test", ["test1", "test2"])


class TestUnitQualifier:
    def test_qualifier(self):
        assert Qualifier("Negation", "AFFIRMED", ordinal=0)
        assert Qualifier("Negation", "NEGATED", ordinal=1)
        assert Qualifier("Negation", "NEGATED", ordinal=1, prob=1)

    def test_qualifier_str(self):
        assert str(Qualifier("Negation", "NEGATED", ordinal=1)) == "Negation.NEGATED"

    def test_qualifier_dict(self):
        assert Qualifier("Negation", "NEGATED", ordinal=1).to_dict() == {"name": "Negation", "value": "NEGATED", "prob": None}
        assert Qualifier("Negation", "NEGATED", ordinal=1, prob=0.8).to_dict() == {
            "name": "Negation", "value": "NEGATED",
            "prob": 0.8,
        }

    def test_compare_equality(self):
        assert Qualifier("Negation", "NEGATED", ordinal=1) == Qualifier("Negation", "NEGATED", ordinal=1)
        assert Qualifier("Negation", "NEGATED", ordinal=1) == Qualifier("Negation", "NEGATED", ordinal=1, prob=0.8)
        assert Qualifier("Negation", "NEGATED", ordinal=1) != Qualifier("Negation", "AFFIRMED", ordinal=0)

    def test_hash_in_set(self):
        qualifiers = {Qualifier("Negation", "AFFIRMED", ordinal=0, prob=1)}

        assert Qualifier("Negation", "AFFIRMED", ordinal=0, prob=0.5) in qualifiers
        assert Qualifier("Negation", "NEGATED", ordinal=1, prob=0.5) not in qualifiers
        assert Qualifier("Temporality", "HISTORICAL", ordinal=1, prob=0.5) not in qualifiers

    def test_spacy_has_extension(self):
        assert Span.has_extension(ATTR_QUALIFIERS)
        assert Span.has_extension(ATTR_QUALIFIERS_STR)
        assert Span.has_extension(ATTR_QUALIFIERS_DICT)

    def test_spacy_extension_default(self, nlp):
        doc = nlp("dit is een test")
        assert getattr(doc[0:3]._, ATTR_QUALIFIERS) is None


class TestUnitQualifierFactory:
    def test_create_factory(self):
        assert QualifierFactory("Negation", ["AFFIRMED", "NEGATED"])

    def test_use_factory(self):
        factory = QualifierFactory("Negation", ["AFFIRMED", "NEGATED"])

        assert factory.get_qualifier(value="AFFIRMED").ordinal == 0
        assert factory.get_qualifier(value="NEGATED").ordinal == 1
        assert factory.get_qualifier(value="AFFIRMED") == Qualifier("Negation", "AFFIRMED", ordinal=0)

    def test_use_factory_unhappy(self):
        factory = QualifierFactory("Negation", ["AFFIRMED", "NEGATED"])

        with pytest.raises(ValueError):
            _ = factory.get_qualifier(value="UNKNOWN")


class TestUnitQualifierDetector:
    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_create_qualifier_detector(self):
        _ = QualifierDetector()

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_initialize_qualifiers(self, entity):
        qd = QualifierDetector()

        qd._initialize_qualifiers(entity)

        assert getattr(entity._, ATTR_QUALIFIERS) == set()

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_no_init(self, entity, mock_factory):
        qd = QualifierDetector()
        qualifier = mock_factory.get_qualifier("test1")

        qd.add_qualifier_to_ent(entity, qualifier)

        assert len(getattr(entity._, ATTR_QUALIFIERS)) == 1
        assert str(mock_factory.get_qualifier("test1")) in getattr(entity._, ATTR_QUALIFIERS_STR)
        assert {"label": "test.test1", "prob": None} in getattr(entity._, ATTR_QUALIFIERS_DICT)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_with_init(self, entity, mock_factory):
        qd = QualifierDetector()
        qualifier = mock_factory.get_qualifier("test1")

        qd._initialize_qualifiers(entity)
        qd.add_qualifier_to_ent(entity, qualifier)

        assert len(getattr(entity._, ATTR_QUALIFIERS)) == 1
        assert str(mock_factory.get_qualifier("test1")) in getattr(entity._, ATTR_QUALIFIERS_STR)
        assert {"label": "test.test1", "prob": None} in getattr(entity._, ATTR_QUALIFIERS_DICT)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_multiple(self, entity, mock_factory):
        qd = QualifierDetector()
        qd._initialize_qualifiers(entity)
        qualifier_1 = mock_factory.get_qualifier("test1")
        qualifier_2 = mock_factory.get_qualifier("test2")

        qd.add_qualifier_to_ent(entity, qualifier_1)
        qd.add_qualifier_to_ent(entity, qualifier_2)

        assert len(getattr(entity._, ATTR_QUALIFIERS)) == 2
        assert str(mock_factory.get_qualifier("test1")) in getattr(entity._, ATTR_QUALIFIERS_STR)
        assert str(mock_factory.get_qualifier("test2")) in getattr(entity._, ATTR_QUALIFIERS_STR)
        assert {"label": "test.test1", "prob": None} in getattr(entity._, ATTR_QUALIFIERS_DICT)
        assert {"label": "test.test2", "prob": None} in getattr(entity._, ATTR_QUALIFIERS_DICT)
