import importlib.util

from .context_algorithm import ContextAlgorithm, ContextRule, ContextRuleDirection
from .qualifier import (
    ATTR_QUALIFIERS,
    Qualifier,
    QualifierDetector,
    QualifierFactory,
    get_qualifiers,
    set_qualifiers,
)

if importlib.util.find_spec("transformers") is not None:
    from .transformer import NegationTransformer
