from unittest.mock import patch

import pytest
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
def entity(nlp):
    doc = nlp("dit is een test")
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

    @pytest.mark.parametrize(
        "q1_kwargs, q2_kwargs, expected_equality",
        [
            ({}, {}, True),
            ({}, {"prob": 0.8}, True),
            ({}, {"value": "Affirmed", "is_default": False}, False),
        ],
    )
    def test_compare_equality(self, q1_kwargs, q2_kwargs, expected_equality):
        # Arrange
        kwargs = {"name": "Negation", "value": "Negated", "is_default": True}
        q1 = Qualifier(**(kwargs | q1_kwargs))
        q2 = Qualifier(**(kwargs | q2_kwargs))

        # Act
        equality = q1 == q2

        # Assert
        assert equality == expected_equality

    @pytest.mark.parametrize(
        "qualifier_kwargs, expected_in_set",
        [
            ({"prob": 0.5}, True),
            ({"value": "Negated", "is_default": True, "prob": 0.5}, False),
            (
                {
                    "name": "Temporality",
                    "value": "HISTORICAL",
                    "is_default": True,
                    "prob": 0.5,
                },
                False,
            ),
        ],
    )
    def test_hash_in_set(self, qualifier_kwargs, expected_in_set):
        # Arrange
        kwargs = {
            "name": "Negation",
            "value": "Affirmed",
            "is_default": False,
            "prob": 1,
        }
        qualifiers = {Qualifier(**kwargs)}

        # Act
        in_set = Qualifier(**(kwargs | qualifier_kwargs)) in qualifiers

        # Assert
        assert in_set == expected_in_set

    @pytest.mark.parametrize(
        "extension, expected_has_extension",
        [
            (ATTR_QUALIFIERS, True),
            (ATTR_QUALIFIERS_STR, True),
            (ATTR_QUALIFIERS_DICT, True),
        ],
    )
    def test_spacy_has_extension(self, extension, expected_has_extension):
        # Act
        has_extension = Span.has_extension(extension)

        # Assert
        assert has_extension == expected_has_extension

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
    @pytest.mark.parametrize(
        "value, expected_is_default",
        [
            ("Affirmed", True),
            ("Negated", False),
        ],
    )
    def test_use_factory(self, value, expected_is_default):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"])

        # Act
        is_default = factory.create(value=value).is_default

        # Assert
        assert is_default == expected_is_default

    @pytest.mark.parametrize(
        "value, expected_is_default",
        [
            ("Affirmed", False),
            ("Negated", True),
        ],
    )
    def test_use_factory_nondefault(self, value, expected_is_default):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"], default="Negated")

        # Act
        is_default = factory.create(value=value).is_default

        # Assert
        assert is_default == expected_is_default

    @pytest.mark.parametrize(
        "value, expected_priority",
        [
            ("Absent", 0),
            ("Uncertain", 1),
            ("Present", 2),
        ],
    )
    def test_use_factory_priority_default(self, value, expected_priority):
        # Arrange
        factory = QualifierClass(
            "Presence", ["Absent", "Uncertain", "Present"], default="Present"
        )

        # Act
        priority = factory.create(value=value).priority

        # Assert
        assert priority == expected_priority

    @pytest.mark.parametrize(
        "value, expected_priority",
        [
            ("Absent", 1),
            ("Uncertain", 100),
            ("Present", 0),
        ],
    )
    def test_use_factory_priority_nondefault(self, value, expected_priority):
        # Arrange
        factory = QualifierClass(
            "Presence",
            ["Absent", "Uncertain", "Present"],
            default="Present",
            priorities={"Absent": 1, "Uncertain": 100, "Present": 0},
        )

        # Act
        priority = factory.create(value=value).priority

        # Assert
        assert priority == expected_priority

    def test_use_factory_unhappy(self):
        # Arrange
        factory = QualifierClass("Negation", ["Affirmed", "Negated"])

        # Assert
        with pytest.raises(ValueError):
            # Act
            _ = factory.create(value="Unknown")


class TestUnitQualifierDetector:
    @patch("clinlp.ie.qualifier.qualifier.QualifierDetector.__abstractmethods__", set())
    def test_add_qualifier_no_init(self, entity, mock_factory):
        # Arrange
        qd = QualifierDetector()
        qualifier = mock_factory.create("test1")

        # Assert
        with pytest.raises(RuntimeError):
            # Act
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
