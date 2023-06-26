from spacy.tokens import Span

from .qualifier import Qualifier, QualifierDetector
from .rule_based import (
    QUALIFIERS_ATTR,
    ContextMatcher,
    ContextRule,
    ContextRuleDirection,
)

Span.set_extension(name=QUALIFIERS_ATTR, default=None)
