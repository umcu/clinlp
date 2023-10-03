from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Optional

from spacy.tokens import Doc, Span

ATTR_QUALIFIERS = "qualifiers"
ATTR_QUALIFIERS_STR = f"{ATTR_QUALIFIERS}_str"
ATTR_QUALIFIERS_DICT = f"{ATTR_QUALIFIERS}_dict"


def qualifiers_to_str(ent: Span) -> Optional[set[str]]:
    qualifiers = getattr(getattr(ent, "_"), ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return {str(q) for q in qualifiers}


def qualifiers_to_dict(ent: Span) -> Optional[list[dict]]:
    qualifiers = getattr(getattr(ent, "_"), ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return [q.to_dict() for q in qualifiers]


Span.set_extension(name=ATTR_QUALIFIERS, default=None)
Span.set_extension(name=ATTR_QUALIFIERS_STR, getter=qualifiers_to_str)
Span.set_extension(name=ATTR_QUALIFIERS_DICT, getter=qualifiers_to_dict)


def get_qualifiers(entity: Span) -> set["Qualifier"]:
    return getattr(getattr(entity, "_"), ATTR_QUALIFIERS)


def set_qualifiers(entity: Span, qualifiers: set["Qualifier"]) -> None:
    setattr(getattr(entity, "_"), ATTR_QUALIFIERS, qualifiers)


@dataclass(frozen=True)
class Qualifier:
    name: str = field(compare=True)
    value: str = field(compare=True)
    is_default: bool = field(compare=True)
    prob: Optional[float] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "is_default": self.is_default,
            "prob": self.prob,
        }

    def __str__(self) -> str:
        return f"{self.name}.{self.value}"


class QualifierFactory:
    def __init__(self, name: str, values: list[str], default: Optional[str] = None):
        self.name = name
        self.default = default or values[0]

        if len(set(values)) != len(values):
            raise ValueError(f"Please do not provide any duplicate values ({values})")

        if self.default not in values:
            raise ValueError(f"Default {default} not in provided value {values}")

        self.values = values

    def create(self, value: Optional[str] = None, **kwargs) -> Qualifier:
        if value is None:
            value = self.default

        if value not in self.values:
            raise ValueError(
                f"The qualifier {self.name} cannot take value '{value}'. Please choose one of {self.values}."
            )

        is_default = value == self.default

        return Qualifier(name=self.name, value=value, is_default=is_default, **kwargs)


class QualifierDetector(ABC):
    """For usage as a spaCy pipeline component"""

    @property
    @abstractmethod
    def qualifier_factories(self) -> dict[str, QualifierFactory]:
        pass

    @abstractmethod
    def _detect_qualifiers(self, doc: Doc) -> None:
        pass

    @staticmethod
    def add_qualifier_to_ent(entity: Span, new_qualifier: Qualifier) -> None:
        qualifiers = get_qualifiers(entity)

        if qualifiers is None:
            raise RuntimeError(
                "Cannot add qualifier to entity with non-initialized qualifiers."
            )

        try:
            old_qualifier = next(
                iter(q for q in qualifiers if q.name == new_qualifier.name)
            )

            qualifiers.remove(old_qualifier)
            qualifiers.add(new_qualifier)

        except StopIteration:
            qualifiers.add(new_qualifier)

        set_qualifiers(entity, qualifiers)

    def _initialize_ent_qualifiers(self, entity: Span) -> None:
        if get_qualifiers(entity) is None:
            set_qualifiers(entity, set())

        for _, factory in self.qualifier_factories.items():
            self.add_qualifier_to_ent(entity, factory.create())

    def __call__(self, doc: Doc) -> Doc:
        if len(doc.ents) == 0:
            return doc

        for ent in doc.ents:
            self._initialize_ent_qualifiers(ent)

        self._detect_qualifiers(doc)

        return doc
