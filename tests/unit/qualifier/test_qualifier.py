from unittest.mock import patch

import pytest
import spacy
from spacy.tokens import Span

import clinlp  # noqa
from clinlp.qualifier import (
    Qualifier,
    QualifierDetector,
    QualifierFactory,
    get_qualifiers,
    set_qualifiers,
)
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


@pytest.fixture
def mock_factory_2():
    return QualifierFactory("test2", ["abc", "def"])


class TestUnitQualifier:
    def test_qualifier(self):
        assert Qualifier("Negation", "Affirmed", is_default=False)
        assert Qualifier("Negation", "Negated", is_default=True)
        assert Qualifier("Negation", "Negated", is_default=True, prob=1)

    def test_qualifier_str(self):
        assert (
            str(Qualifier("Negation", "Negated", is_default=True)) == "Negation.Negated"
        )

    def test_qualifier_dict(self):
        assert Qualifier("Negation", "Negated", is_default=True).to_dict() == {
            "name": "Negation",
            "value": "Negated",
            "is_default": True,
            "prob": None,
        }
        assert Qualifier(
            "Negation", "Negated", is_default=True, prob=0.8
        ).to_dict() == {
            "name": "Negation",
            "value": "Negated",
            "is_default": True,
            "prob": 0.8,
        }

    def test_compare_equality(self):
        assert Qualifier("Negation", "Negated", is_default=True) == Qualifier(
            "Negation", "Negated", is_default=True
        )
        assert Qualifier("Negation", "Negated", is_default=True) == Qualifier(
            "Negation", "Negated", is_default=True, prob=0.8
        )
        assert Qualifier("Negation", "Negated", is_default=True) != Qualifier(
            "Negation", "Affirmed", is_default=False
        )

    def test_hash_in_set(self):
        qualifiers = {Qualifier("Negation", "Affirmed", is_default=False, prob=1)}

        assert (
            Qualifier("Negation", "Affirmed", is_default=False, prob=0.5) in qualifiers
        )
        assert (
            Qualifier("Negation", "Negated", is_default=True, prob=0.5)
            not in qualifiers
        )
        assert (
            Qualifier("Temporality", "HISTORICAL", is_default=True, prob=0.5)
            not in qualifiers
        )

    def test_spacy_has_extension(self):
        assert Span.has_extension(ATTR_QUALIFIERS)
        assert Span.has_extension(ATTR_QUALIFIERS_STR)
        assert Span.has_extension(ATTR_QUALIFIERS_DICT)

    def test_spacy_extension_default(self, nlp):
        doc = nlp("dit is een test")
        assert get_qualifiers(doc[0:3]) is None

    def test_set_qualifiers(self, mock_factory, entity):
        qualifiers = {mock_factory.create()}
        set_qualifiers(entity, qualifiers)

        assert get_qualifiers(entity) == qualifiers


class TestUnitQualifierFactory:
    def test_create_factory(self):
        assert QualifierFactory("Negation", ["Affirmed", "Negated"])

    def test_use_factory(self):
        factory = QualifierFactory("Negation", ["Affirmed", "Negated"])

        assert factory.create(value="Affirmed").is_default
        assert not factory.create(value="Negated").is_default
        assert factory.create(value="Affirmed") == Qualifier(
            "Negation", "Affirmed", is_default=True
        )

    def test_use_factory_nondefault(self):
        factory = QualifierFactory(
            "Negation", ["Affirmed", "Negated"], default="Negated"
        )

        assert not factory.create(value="Affirmed").is_default
        assert factory.create(value="Negated").is_default
        assert factory.create(value="Affirmed") == Qualifier(
            "Negation", "Affirmed", is_default=False
        )

    def test_use_factory_unhappy(self):
        factory = QualifierFactory("Negation", ["Affirmed", "Negated"])

        with pytest.raises(ValueError):
            _ = factory.create(value="Unknown")


class TestUnitQualifierDetector:
    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_create_qualifier_detector(self):
        _ = QualifierDetector()

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_no_init(self, entity, mock_factory):
        qd = QualifierDetector()
        qualifier = mock_factory.create("test1")

        with pytest.raises(RuntimeError):
            qd.add_qualifier_to_ent(entity, qualifier)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_default(self, entity, mock_factory):
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier = mock_factory.create("test1")

        with patch(
            "clinlp.qualifier.qualifier.QualifierDetector.qualifier_factories",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        qd.add_qualifier_to_ent(entity, qualifier)

        assert len(get_qualifiers(entity)) == 1
        assert qualifier in get_qualifiers(entity)
        assert mock_factory.create("test2") not in get_qualifiers(entity)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_non_default(self, entity, mock_factory):
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier = mock_factory.create("test2")

        with patch(
            "clinlp.qualifier.qualifier.QualifierDetector.qualifier_factories",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        qd.add_qualifier_to_ent(entity, qualifier)

        assert len(get_qualifiers(entity)) == 1
        assert mock_factory.create("test1") not in get_qualifiers(entity)
        assert qualifier in get_qualifiers(entity)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_overwrite_nondefault(self, entity, mock_factory):
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier_1 = mock_factory.create("test1")
        qualifier_2 = mock_factory.create("test2")

        with patch(
            "clinlp.qualifier.qualifier.QualifierDetector.qualifier_factories",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        qd.add_qualifier_to_ent(entity, qualifier_2)
        qd.add_qualifier_to_ent(entity, qualifier_1)

        assert len(get_qualifiers(entity)) == 1
        assert qualifier_1 in get_qualifiers(entity)
        assert qualifier_2 not in get_qualifiers(entity)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_multiple(self, entity, mock_factory, mock_factory_2):
        qd = QualifierDetector()
        qualifier_1 = mock_factory.create("test2")
        qualifier_2 = mock_factory_2.create("abc")

        factories = {"test1": mock_factory, "test2": mock_factory_2}

        with patch(
            "clinlp.qualifier.qualifier.QualifierDetector.qualifier_factories",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        qd.add_qualifier_to_ent(entity, qualifier_1)
        qd.add_qualifier_to_ent(entity, qualifier_2)

        assert len(get_qualifiers(entity)) == 2
        assert qualifier_1 in get_qualifiers(entity)
        assert qualifier_2 in get_qualifiers(entity)

    @patch("clinlp.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_initialize_qualifiers(self, entity, mock_factory, mock_factory_2):
        qd = QualifierDetector()

        factories = {"test1": mock_factory, "test2": mock_factory_2}

        with patch(
            "clinlp.qualifier.qualifier.QualifierDetector.qualifier_factories",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        assert len(get_qualifiers(entity)) == 2
        assert mock_factory.create() in get_qualifiers(entity)
        assert mock_factory.create("test2") not in get_qualifiers(entity)
        assert mock_factory_2.create() in get_qualifiers(entity)
        assert mock_factory_2.create("def") not in get_qualifiers(entity)
