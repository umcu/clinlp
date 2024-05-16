from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from spacy.tokens import Doc, Span

ATTR_QUALIFIERS = "qualifiers"
ATTR_QUALIFIERS_STR = f"{ATTR_QUALIFIERS}_str"
ATTR_QUALIFIERS_DICT = f"{ATTR_QUALIFIERS}_dict"


def qualifiers_to_str(ent: Span) -> Optional[set[str]]:
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return {str(q) for q in qualifiers}


def qualifiers_to_dict(ent: Span) -> Optional[list[dict]]:
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return [q.to_dict() for q in qualifiers]


Span.set_extension(name=ATTR_QUALIFIERS, default=None)
Span.set_extension(name=ATTR_QUALIFIERS_STR, getter=qualifiers_to_str)
Span.set_extension(name=ATTR_QUALIFIERS_DICT, getter=qualifiers_to_dict)


def get_qualifiers(entity: Span) -> set["Qualifier"]:
    return getattr(entity._, ATTR_QUALIFIERS)


def set_qualifiers(entity: Span, qualifiers: set["Qualifier"]) -> None:
    setattr(entity._, ATTR_QUALIFIERS, qualifiers)


@dataclass(frozen=True)
class Qualifier:
    name: str = field(compare=True)
    value: str = field(compare=True)
    is_default: bool = field(compare=True)
    priority: int = field(default=0, compare=False)
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


class QualifierClass:
    def __init__(
        self,
        name: str,
        values: list[str],
        default: Optional[str] = None,
        priorities: Optional[dict] = None,
    ):
        self.name = name
        self.values = values
        self.default = default or values[0]
        self.priorities = priorities or {value: n for n, value in enumerate(values)}

        if len(set(values)) != len(values):
            raise ValueError(f"Please do not provide any duplicate values ({values})")

        if self.default not in values:
            raise ValueError(f"Default {default} not in provided value {values}")

    def create(self, value: Optional[str] = None, **kwargs) -> Qualifier:
        if value is None:
            value = self.default

        if value not in self.values:
            raise ValueError(
                f"The qualifier {self.name} cannot take value '{value}'. "
                f"Please choose one of {self.values}."
            )

        is_default = value == self.default
        priority = self.priorities[value]

        return Qualifier(
            name=self.name,
            value=value,
            is_default=is_default,
            priority=priority,
            **kwargs,
        )


class QualifierDetector(ABC):
    """For usage as a spaCy pipeline component"""

    @property
    @abstractmethod
    def qualifier_classes(self) -> dict[str, QualifierClass]:
        pass

    @staticmethod
    def add_qualifier_to_ent(entity: Span, new_qualifier: Qualifier) -> None:
        qualifiers = get_qualifiers(entity)

        if qualifiers is None:
            raise RuntimeError(
                "Cannot add qualifier to entity with non-initialized qualifiers."
            )

        qualifiers = {q for q in qualifiers if q.name != new_qualifier.name}
        qualifiers.add(new_qualifier)

        set_qualifiers(entity, qualifiers)

    def _initialize_ent_qualifiers(self, entity: Span) -> None:
        if get_qualifiers(entity) is None:
            set_qualifiers(entity, set())

        for _, qualifier_class in self.qualifier_classes.items():
            self.add_qualifier_to_ent(entity, qualifier_class.create())

    @abstractmethod
    def _detect_qualifiers(self, doc: Doc) -> None:
        pass

    def __call__(self, doc: Doc) -> Doc:
        if len(doc.ents) == 0:
            return doc

        for ent in doc.ents:
            self._initialize_ent_qualifiers(ent)

        self._detect_qualifiers(doc)

        return doc
