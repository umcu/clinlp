"""Reusable components for detecting qualifiers in clinical text."""

from abc import abstractmethod
from dataclasses import dataclass, field

from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span

from clinlp.ie import SPANS_KEY

ATTR_QUALIFIERS = "qualifiers"
ATTR_QUALIFIERS_STR = f"{ATTR_QUALIFIERS}_str"
ATTR_QUALIFIERS_DICT = f"{ATTR_QUALIFIERS}_dict"


def qualifiers_to_str(ent: Span) -> set[str] | None:
    """
    Get qualifier information in string format.

    Parameters
    ----------
    ent
        The entity to get qualifiers from.

    Returns
    -------
    ``Optional[set[str]]``
        The qualifiers in string format, e.g. ``{'Presence.Present', ...}``, or ``None``
        if no qualifiers are present.
    """
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return {str(q) for q in qualifiers}


def qualifiers_to_dict(ent: Span) -> list[dict] | None:
    """
    Get qualifier information in dictionary format.

    Parameters
    ----------
    ent
        The entity to get qualifiers from.

    Returns
    -------
    ``Optional[list[dict]]``
        The qualifiers in ``dict`` format, e.g.
        ``[{'Name': 'Presence', 'Value': 'Present', 'is_default': True}, ...]``,
        or ``None`` if no qualifiers are present.
    """
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return [q.to_dict() for q in qualifiers]


Span.set_extension(name=ATTR_QUALIFIERS, default=None)
Span.set_extension(name=ATTR_QUALIFIERS_STR, getter=qualifiers_to_str)
Span.set_extension(name=ATTR_QUALIFIERS_DICT, getter=qualifiers_to_dict)


def get_qualifiers(entity: Span) -> set["Qualifier"]:
    """
    Get the qualifiers for an entity.

    Returns
    -------
    ``set[Qualifier]``
        The qualifiers.
    """
    return getattr(entity._, ATTR_QUALIFIERS)


def set_qualifiers(entity: Span, qualifiers: set["Qualifier"]) -> None:
    """
    Set the qualifiers for an entity.

    Parameters
    ----------
    entity
        The entity to set qualifiers for.
    qualifiers
        The qualifiers to set.
    """
    setattr(entity._, ATTR_QUALIFIERS, qualifiers)


@dataclass(frozen=True)
class Qualifier:
    """
    A qualifier for an entity.

    A qualifier is a piece of information that provides additional context to an entity.
    For example, a ``Presence`` qualifier with a value of ``Present`` or ``Absent``. A
    qualifier has a fixed value.
    """

    name: str = field(compare=True)
    """The name of the qualifier."""

    value: str = field(compare=True)
    """The value of the qualifier."""

    is_default: bool | None = field(default=None, compare=False)
    """Whether the value is the default value."""

    priority: int = field(default=0, compare=False)
    """The priority of the qualifier."""

    prob: float | None = field(default=None, compare=False)
    """The probability of the qualifier."""

    def to_dict(self) -> dict:
        """
        Convert the qualifier to a dictionary.

        Returns
        -------
        ``dict``
            The qualifier as a dictionary.
        """
        return {
            "name": self.name,
            "value": self.value,
            "is_default": self.is_default,
            "prob": self.prob,
        }

    def __str__(self) -> str:
        """
        Get the string representation of the qualifier.

        Returns
        -------
        ``str``
            The string representation of the qualifier.
        """
        return f"{self.name}.{self.value}"


class QualifierClass:
    """
    A qualifier class.

    A qualifier class defines the set of possible values a qualifier can take on. For
    example: ``Presence`` with values ``Present`` and ``Absent``. The qualifier class
    creates qualifiers, although they can also be created directly.
    """

    def __init__(
        self,
        name: str,
        values: list[str],
        default: str | None = None,
        priorities: dict | None = None,
    ) -> None:
        """
        Initialize a qualifier class.

        Parameters
        ----------
        name
            The name of the qualifier.
        values
            The possible values of the qualifier.
        default
            The default value of the qualifier.
        priorities
            The priorities of the values. If not provided, the order of the values is
            used.

        Raises
        ------
        ValueError
            If there are duplicate values.
        ValueError
            If the default value is not in the provided values.
        """
        self.name = name
        self.values = values
        self.default = default or values[0]
        self.priorities = priorities or {value: n for n, value in enumerate(values)}

        if len(set(values)) != len(values):
            msg = f"Please do not provide any duplicate values ({values})."
            raise ValueError(msg)

        if self.default not in values:
            msg = f"Default {default} not in provided value {values}."
            raise ValueError(msg)

    def create(self, value: str | None = None, **kwargs) -> Qualifier:
        """
        Create a qualifier in this qualifier class.

        Parameters
        ----------
        value
            The value for the qualifier.

        Returns
        -------
        ``Qualifier``
            The created qualifier.

        Raises
        ------
        ValueError
            If the value is not in the possible values.
        """
        if value is None:
            value = self.default

        if value not in self.values:
            msg = (
                f"The qualifier {self.name} cannot take value '{value}'. "
                f"Please choose one of {self.values}."
            )
            raise ValueError(msg)

        is_default = value == self.default
        priority = self.priorities[value]

        return Qualifier(
            name=self.name,
            value=value,
            is_default=is_default,
            priority=priority,
            **kwargs,
        )


class QualifierDetector(Pipe):
    """Abstract pipeline component for detecting qualifiers in clinical text."""

    def __init__(self, spans_key: str = SPANS_KEY) -> None:
        """
        Initialize a qualifier detector.

        Parameters
        ----------
        spans_key
            The key for the spans in the ``Doc`` object.
        """
        self.spans_key = spans_key

    @property
    @abstractmethod
    def qualifier_classes(self) -> dict[str, QualifierClass]:
        """
        Obtain the qualifier classes that a ``QualifierDetector`` initializes.

        These are used to initialize the default qualifiers for each entity.

        Returns
        -------
        ``dict[str, QualifierClass]``
            The qualifier classes.
        """

    @staticmethod
    def add_qualifier_to_ent(entity: Span, new_qualifier: Qualifier) -> None:
        """
        Add a qualifier to an entity.

        Preferably, qualifiers should not be added in another way than through this
        method, to ensure consistency.

        Parameters
        ----------
        entity
            The entity to add the qualifier to.
        new_qualifier
            The qualifier to add.

        Raises
        ------
        RuntimeError
            If the entity does not have initialized qualifiers.
        """
        qualifiers = get_qualifiers(entity)

        if qualifiers is None:
            msg = "Cannot add qualifier to entity with non-initialized qualifiers."
            raise RuntimeError(msg)

        qualifiers = {q for q in qualifiers if q.name != new_qualifier.name}
        qualifiers.add(new_qualifier)

        set_qualifiers(entity, qualifiers)

    def _initialize_ent_qualifiers(self, entity: Span) -> None:
        """
        Initialize the qualifiers for an entity.

        Parameters
        ----------
        entity
            The entity to initialize qualifiers for.
        """
        if get_qualifiers(entity) is None:
            set_qualifiers(entity, set())

        for qualifier_class in self.qualifier_classes.values():
            self.add_qualifier_to_ent(entity, qualifier_class.create())

    @abstractmethod
    def _detect_qualifiers(self, doc: Doc) -> None:
        """
        Detect qualifiers for the entities in a document.

        Parameters
        ----------
        doc
            The document to process.
        """

    def __call__(self, doc: Doc) -> Doc:
        """
        Initialize default qualifiers and run detection.

        Parameters
        ----------
        doc
            The document to process.

        Returns
        -------
        ``Doc``
            The processed document.
        """
        if self.spans_key not in doc.spans or len(doc.spans[self.spans_key]) == 0:
            return doc

        for ent in doc.spans[self.spans_key]:
            self._initialize_ent_qualifiers(ent)

        self._detect_qualifiers(doc)

        return doc
