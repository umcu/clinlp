"""Information Extraction (IE) module for ``clinlp``."""

from .entity import SPANS_KEY, RuleBasedEntityMatcher
from .term import Term

__all__ = ["SPANS_KEY", "RuleBasedEntityMatcher", "Term"]
