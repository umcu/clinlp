import importlib.util

from .context_algorithm import ContextAlgorithm, ContextRule, ContextRuleDirection
from .qualifier import (
    ATTR_QUALIFIERS,
    Qualifier,
    QualifierClass,
    QualifierDetector,
    get_qualifiers,
    set_qualifiers,
)

if importlib.util.find_spec("transformers") is not None:
    from .transformer import (
        ExperiencerTransformer,
        NegationTransformer,
        QualifierTransformer,
    )

__all__ = [
    "ContextAlgorithm",
    "ContextRule",
    "ContextRuleDirection",
    "ATTR_QUALIFIERS",
    "Qualifier",
    "QualifierClass",
    "QualifierDetector",
    "get_qualifiers",
    "set_qualifiers",
    "ExperiencerTransformer",
    "NegationTransformer",
    "QualifierTransformer",
]
