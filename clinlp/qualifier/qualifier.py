from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from spacy.tokens import Doc, Span

ATTR_QUALIFIERS = "qualifiers"
ATTR_QUALIFIERS_STR = f"{ATTR_QUALIFIERS}_str"
ATTR_QUALIFIERS_DICT = f"{ATTR_QUALIFIERS}_dict"


def qualifiers_to_str(ent: Span):
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return {str(q) for q in qualifiers}


def qualifiers_to_dict(ent: Span):
    qualifiers = getattr(ent._, ATTR_QUALIFIERS)

    if qualifiers is None:
        return None

    return [q.to_dict() for q in qualifiers]


Span.set_extension(name=ATTR_QUALIFIERS, default=None)
Span.set_extension(name=ATTR_QUALIFIERS_STR, getter=qualifiers_to_str)
Span.set_extension(name=ATTR_QUALIFIERS_DICT, getter=qualifiers_to_dict)


@dataclass(frozen=True)
class Qualifier:
    name: str = field(compare=True)
    value: str = field(compare=True)
    ordinal: int = field(compare=False)
    prob: Optional[float] = field(default=None, compare=False)

    def to_dict(self):
        return {"label": str(self), "prob": self.prob}

    def __str__(self):
        return f"{self.name}.{self.value}"


class QualifierFactory:
    def __init__(self, name: str, values: list[str]):
        self.name = name

        if len(set(values)) != len(values):
            raise ValueError(f"Please do not provide any duplicate values ({values})")

        self.values = values

    def get_qualifier(self, value: Optional[str] = None, **kwargs):
        if value is None:
            value = self.values[0]

        if value not in self.values:
            raise ValueError(
                f"The qualifier {self.name} cannot take value '{value}'. Please choose one of {self.values}.d"
            )

        return Qualifier(name=self.name, value=value, ordinal=self.values.index(value), **kwargs)


class QualifierDetector(ABC):
    """For usage as a spaCy pipeline component"""

    def _initialize_qualifiers(self, entity: Span):
        setattr(entity._, ATTR_QUALIFIERS, set())

    def add_qualifier_to_ent(self, entity: Span, new_qualifier: Qualifier):
        if getattr(entity._, ATTR_QUALIFIERS) is None:
            self._initialize_qualifiers(entity)

        getattr(entity._, ATTR_QUALIFIERS).add(new_qualifier)

    @abstractmethod
    def detect_qualifiers(self, doc: Doc):
        pass

    def __call__(self, doc: Doc):
        if len(doc.ents) == 0:
            return doc

        for ent in doc.ents:
            self._initialize_qualifiers(ent)

        return self.detect_qualifiers(doc)
