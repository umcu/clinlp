from abc import ABC, abstractmethod
from enum import Enum

from spacy.tokens import Doc


class Qualifier(Enum):
    """
    A qualifier modifies an entity (e.g. negation, temporality, plausibility, etc.).
    """

    pass


class QualifierDetector(ABC):
    @abstractmethod
    def __call__(self, doc: Doc):
        pass
