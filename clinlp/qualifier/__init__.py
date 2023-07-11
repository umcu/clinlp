import importlib.util

from .context_algorithm import ContextAlgorithm, ContextRule, ContextRuleDirection
from .qualifier import QUALIFIERS_ATTR, Qualifier, QualifierDetector

if importlib.util.find_spec("transformers") is not None:
    from .transformer import NegationTransformer
