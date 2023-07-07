from abc import ABC, abstractmethod
from enum import Enum

from spacy.tokens import Doc, Span, Token

QUALIFIERS_ATTR = "qualifiers"

Span.set_extension(name=QUALIFIERS_ATTR, default=None)


class Qualifier(Enum):
    """
    A qualifier modifies an entity (e.g. negation, temporality, plausibility, etc.).
    """

class QualifierDetector(ABC):
    def _initialize_qualifiers(self, entity: Span):
        if getattr(entity._, QUALIFIERS_ATTR) is None:
            setattr(entity._, QUALIFIERS_ATTR, [])

    def add_qualifier_to_ent(self, entity: Span, new_qualifier: Qualifier, prob: float):
        self._initialize_qualifiers(entity)

        getattr(entity._, QUALIFIERS_ATTR).append({'label': str(new_qualifier), 'proba': prob})

    @abstractmethod
    def detect_qualifiers(self, doc: Doc):
        pass

    def __call__(self, doc: Doc):
        if len(doc.ents) == 0:
            return doc

        for ent in doc.ents:
            self._initialize_qualifiers(ent)

        return self.detect_qualifiers(doc)
