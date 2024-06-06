"""Information Extraction (IE) module for ``clinlp``."""

from .entity import SPANS_KEY, RuleBasedEntityMatcher, create_concept_dict
from .term import Term

__all__ = ["SPANS_KEY", "RuleBasedEntityMatcher", "Term", "create_concept_dict"]
