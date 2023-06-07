from spacy.tokens import Span

from .normalizer import Normalizer
from .qualifier import (
    QUALIFIERS_ATTR,
    ContextMatcher,
    ContextRule,
    ContextRuleDirection,
    Qualifier,
)
from .sentencizer import Sentencizer

Span.set_extension(name=QUALIFIERS_ATTR, default=None)
