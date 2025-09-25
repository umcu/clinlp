"""The ``clinlp`` package, a set of tools for processing Dutch clinical text."""

import importlib
import importlib.util

from .ie.entity import RuleBasedEntityMatcher
from .ie.qualifier import ContextAlgorithm
from .language import Clinlp
from .normalizer import Normalizer
from .sentencizer import Sentencizer

if importlib.util.find_spec("transformers") is not None:
    from .ie.qualifier import ExperiencerTransformer, NegationTransformer


__version__ = importlib.metadata.version(__package__ or __name__)


__all__ = [
    "Clinlp",
    "ContextAlgorithm",
    "ExperiencerTransformer",
    "NegationTransformer",
    "Normalizer",
    "RuleBasedEntityMatcher",
    "Sentencizer",
]
