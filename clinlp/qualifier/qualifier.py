from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from spacy.tokens import Doc, Span

QUALIFIERS_ATTR = "qualifiers"

Span.set_extension(name=QUALIFIERS_ATTR, default=None)


class Qualifier(Enum):
    """
    A qualifier modifies an entity (e.g. negation, temporality, plausibility, etc.).
    """

    pass


class QualifierDetector(ABC):
    def _initialize_qualifiers(self, entity: Span):
        setattr(entity._, QUALIFIERS_ATTR, set())

    def add_qualifier_to_ent(self, entity: Span, new_qualifier: Qualifier):
        if getattr(entity._, QUALIFIERS_ATTR) is None:
            self._initialize_qualifiers(entity)

        getattr(entity._, QUALIFIERS_ATTR).add(str(new_qualifier))

    @abstractmethod
    def __call__(self, doc: Doc):
        pass
