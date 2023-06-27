from unittest.mock import patch

import pytest
import spacy
from spacy.tokens import Span

import clinlp
from clinlp.qualifier import QUALIFIERS_ATTR, Qualifier, QualifierDetector


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


@pytest.fixture
def entity():
    doc = spacy.blank("clinlp")("dit is een test")
    return doc[2:3]


@pytest.fixture
def mock_qualifier():
    return Qualifier("test", ["test1", "test2"])


class TestUnitQualifier:
    def test_qualifier(self):
        q = Qualifier("NEGATION", ["AFFIRMED", "NEGATED"])

        assert q["AFFIRMED"]
        assert q["NEGATED"]

    def test_spacy_has_extension(self):
        assert Span.has_extension(QUALIFIERS_ATTR)

    def test_spacy_extension_default(self, nlp):
        doc = nlp("dit is een test")
        assert getattr(doc[0:3]._, QUALIFIERS_ATTR) is None


class TestUnitQualifierDetector:
    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_create_qualifier_detector(self):
        _ = QualifierDetector()

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_initialize_qualifiers(self, entity):
        qd = QualifierDetector()

        qd._initialize_qualifiers(entity)

        assert getattr(entity._, QUALIFIERS_ATTR) == set()

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_no_init(self, entity, mock_qualifier):
        qd = QualifierDetector()

        qd.add_qualifier_to_ent(entity, mock_qualifier.test1)

        assert len(getattr(entity._, QUALIFIERS_ATTR)) == 1
        assert str(mock_qualifier.test1) in getattr(entity._, QUALIFIERS_ATTR)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_with_init(self, entity, mock_qualifier):
        qd = QualifierDetector()

        qd._initialize_qualifiers(entity)
        qd.add_qualifier_to_ent(entity, mock_qualifier.test1)

        assert len(getattr(entity._, QUALIFIERS_ATTR)) == 1
        assert str(mock_qualifier.test1) in getattr(entity._, QUALIFIERS_ATTR)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_multiple(self, entity, mock_qualifier):
        qd = QualifierDetector()
        qd._initialize_qualifiers(entity)

        qd.add_qualifier_to_ent(entity, mock_qualifier.test1)
        qd.add_qualifier_to_ent(entity, mock_qualifier.test2)

        assert len(getattr(entity._, QUALIFIERS_ATTR)) == 2
        assert str(mock_qualifier.test1) in getattr(entity._, QUALIFIERS_ATTR)
        assert str(mock_qualifier.test2) in getattr(entity._, QUALIFIERS_ATTR)
