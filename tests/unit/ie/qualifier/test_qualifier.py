from unittest.mock import patch

import pytest
import spacy
from spacy.tokens import Span

import clinlp  # noqa
from clinlp.ie.qualifier import (
    ATTR_QUALIFIERS,
    Qualifier,
    QualifierClass,
    QualifierDetector,
    get_qualifiers,
    set_qualifiers,
)
from clinlp.ie.qualifier.qualifier import ATTR_QUALIFIERS_DICT, ATTR_QUALIFIERS_STR


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


@pytest.fixture
def entity():
    doc = spacy.blank("clinlp")("dit is een test")
    return doc[2:3]


@pytest.fixture
def mock_factory():
    return QualifierClass("test", ["test1", "test2"])


@pytest.fixture
def mock_factory_2():
    return QualifierClass("test2", ["abc", "def"])


class TestUnitQualifier:
    def test_qualifier_str(self):
        # Arrange
        qualifier = Qualifier("Negation", "Negated", is_default=True)

        # Act
        qualifier_str = str(qualifier)

        # Assert
        assert qualifier_str == "Negation.Negated"

    def test_qualifier_priority(self):
        # Arrange
        qualifier = Qualifier("Negation", "Negated", is_default=False, priority=10)

        # Act
        priority = qualifier.priority

        # Assert
        assert priority == 10

    def test_qualifier_dict_1(self):
        # Arrange
        qualifier = Qualifier("Negation", "Negated", is_default=True)

        # Act
        qualifier_dict = qualifier.to_dict()

        # Assert
        assert qualifier_dict == {
            "name": "Negation",
            "value": "Negated",
            "is_default": True,
            "prob": None,
        }

    def test_qualifier_dict_2(self):
        # Arrange
        qualifier = Qualifier("Negation", "Negated", is_default=True, prob=0.8)

        # Act
        qualifier_dict = qualifier.to_dict()

        # Assert
        assert qualifier_dict == {
            "name": "Negation",
            "value": "Negated",
            "is_default": True,
            "prob": 0.8,
        }

    def test_compare_equality(self):
        # Arrange, Act & Assert
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
        # Arrange & Act
        qualifiers = {Qualifier("Negation", "Affirmed", is_default=False, prob=1)}

        # Assert
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
        # Arrange, Act & Assert
        assert Span.has_extension(ATTR_QUALIFIERS)
        assert Span.has_extension(ATTR_QUALIFIERS_STR)
        assert Span.has_extension(ATTR_QUALIFIERS_DICT)

    def test_spacy_extension_default(self, nlp):
        # Arrange
        doc = nlp("dit is een test")

        # Act
        qualifiers = get_qualifiers(doc[0:3])

        # Assert
        assert qualifiers is None

    def test_set_qualifiers(self, mock_factory, entity):
        # Arrange
        qualifiers = {mock_factory.create()}

        # Act
        set_qualifiers(entity, qualifiers)

        # Assert
        assert get_qualifiers(entity) == qualifiers


class TestUnitQualifierFactory:
    def test_use_factory(self):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"])

        # Act & Assert
        assert factory.create(value="Affirmed").is_default
        assert not factory.create(value="Negated").is_default
        assert factory.create(value="Affirmed") == Qualifier(
            "Negation", "Affirmed", is_default=True
        )

    def test_use_factory_nondefault(self):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"], default="Negated")

        # Act & Assert
        assert not factory.create(value="Affirmed").is_default
        assert factory.create(value="Negated").is_default
        assert factory.create(value="Affirmed") == Qualifier(
            "Negation", "Affirmed", is_default=False
        )

    def test_use_factory_priority_default(self):
        # Arrange
        factory = QualifierClass(
            "Presence", ["Absent", "Uncertain", "Present"], default="Present"
        )

        # Act & Assert
        assert factory.create("Absent").priority == 0
        assert factory.create("Uncertain").priority == 1
        assert factory.create("Present").priority == 2

    def test_use_factory_priority_nondefault(self):
        # Arrange
        factory = QualifierClass(
            "Presence",
            ["Absent", "Uncertain", "Present"],
            default="Present",
            priorities={"Absent": 1, "Uncertain": 100, "Present": 0},
        )

        # Act & Assert
        assert factory.create("Absent").priority == 1
        assert factory.create("Uncertain").priority == 100
        assert factory.create("Present").priority == 0

    def test_use_factory_unhappy(self):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = factory.create(value="Unknown")


class TestUnitQualifierDetector:
    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_no_init(self, entity, mock_factory):
        # Arrange
        qd = QualifierDetector()
        qualifier = mock_factory.create("test1")

        # Act & Assert
        with pytest.raises(RuntimeError):
            qd.add_qualifier_to_ent(entity, qualifier)

    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_default(self, entity, mock_factory):
        # Arrange
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier = mock_factory.create("test1")

        with patch(
            "clinlp.ie.qualifier.qualifier.QualifierDetector.qualifier_classes",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        # Act
        qd.add_qualifier_to_ent(entity, qualifier)

        # Assert
        assert len(get_qualifiers(entity)) == 1
        assert qualifier in get_qualifiers(entity)
        assert mock_factory.create("test2") not in get_qualifiers(entity)

    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_non_default(self, entity, mock_factory):
        # Arrange
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier = mock_factory.create("test2")

        with patch(
            "clinlp.ie.qualifier.qualifier.QualifierDetector.qualifier_classes",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        # Act
        qd.add_qualifier_to_ent(entity, qualifier)

        # Assert
        assert len(get_qualifiers(entity)) == 1
        assert mock_factory.create("test1") not in get_qualifiers(entity)
        assert qualifier in get_qualifiers(entity)

    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_overwrite_nondefault(self, entity, mock_factory):
        # Arrange
        qd = QualifierDetector()
        factories = {"test": mock_factory}
        qualifier_1 = mock_factory.create("test1")
        qualifier_2 = mock_factory.create("test2")

        with patch(
            "clinlp.ie.qualifier.qualifier.QualifierDetector.qualifier_classes",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        # Act
        qd.add_qualifier_to_ent(entity, qualifier_2)
        qd.add_qualifier_to_ent(entity, qualifier_1)

        # Assert
        assert len(get_qualifiers(entity)) == 1
        assert qualifier_1 in get_qualifiers(entity)
        assert qualifier_2 not in get_qualifiers(entity)

    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_multiple(self, entity, mock_factory, mock_factory_2):
        # Arrange
        qd = QualifierDetector()
        qualifier_1 = mock_factory.create("test2")
        qualifier_2 = mock_factory_2.create("abc")
        factories = {"test1": mock_factory, "test2": mock_factory_2}

        with patch(
            "clinlp.ie.qualifier.qualifier.QualifierDetector.qualifier_classes",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        # Act
        qd.add_qualifier_to_ent(entity, qualifier_1)
        qd.add_qualifier_to_ent(entity, qualifier_2)

        # Assert
        assert len(get_qualifiers(entity)) == 2
        assert qualifier_1 in get_qualifiers(entity)
        assert qualifier_2 in get_qualifiers(entity)

    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_initialize_qualifiers(self, entity, mock_factory, mock_factory_2):
        # Arrange
        qd = QualifierDetector()
        factories = {"test1": mock_factory, "test2": mock_factory_2}

        # Act
        with patch(
            "clinlp.ie.qualifier.qualifier.QualifierDetector.qualifier_classes",
            factories,
        ):
            qd._initialize_ent_qualifiers(entity)

        # Assert
        assert len(get_qualifiers(entity)) == 2
        assert mock_factory.create() in get_qualifiers(entity)
        assert mock_factory.create("test2") not in get_qualifiers(entity)
        assert mock_factory_2.create() in get_qualifiers(entity)
        assert mock_factory_2.create("def") not in get_qualifiers(entity)
