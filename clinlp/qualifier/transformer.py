import statistics
from typing import Callable, Optional, Tuple

import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, RobertaForTokenClassification

from clinlp.qualifier.qualifier import QUALIFIERS_ATTR, Qualifier, QualifierDetector

_defaults_negation_transformer = {
    "token_window": 32,
    "probas_aggregator": statistics.mean,
    "threshold": 0.5,
    "strip_entities": True,
    "placeholder": None,
}

TRANSFORMER_REPO = "UMCU/MedRoBERTa.nl_NegationDetection"


@Language.factory(name="clinlp_negation_transformer", requires=["doc.ents"], assigns=[f"span._.{QUALIFIERS_ATTR}"])
def make_negation_transformer(nlp: Language, name: str, **_defaults_negation_transformer):
    return NegationTransformer(nlp, **_defaults_negation_transformer)


class NegationTransformer(QualifierDetector):
    def __init__(
        self,
        nlp: Language,
        token_window: int = _defaults_negation_transformer["token_window"],
        probas_aggregator: Callable = _defaults_negation_transformer["probas_aggregator"],
        threshold: float = _defaults_negation_transformer["threshold"],
        strip_entities: bool = _defaults_negation_transformer["strip_entities"],
        placeholder: Optional[str] = _defaults_negation_transformer["placeholder"],
        **kwargs,
    ):
        self.nlp = nlp
        self.token_window = token_window
        self.probas_aggregator = probas_aggregator
        self.threshold = threshold
        self.strip_entities = strip_entities
        self.placeholder = placeholder

        self.negation_qualifier = Qualifier("Negation", ["Affirmed", "Negated"])

        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_REPO)
        self.model = RobertaForTokenClassification.from_pretrained(TRANSFORMER_REPO)

        super().__init__(**kwargs)

    @staticmethod
    def _get_ent_window(self, ent: Span, token_window: int) -> Tuple[str, int, int]:
        start_token_i = max(0, ent.start - token_window)
        end_token_i = min(len(ent.doc), ent.end + token_window)

        text_span = ent.doc[start_token_i:end_token_i]

        ent_start_char = ent.start_char - text_span.start_char
        ent_end_char = ent.end_char - text_span.start_char

        return str(text_span), ent_start_char, ent_end_char

    @staticmethod
    def _trim_ent_boundaries(text: str, ent_start_char: int, ent_end_char: int) -> Tuple[str, int, int]:
        entity = text[ent_start_char:ent_end_char]

        ent_start_char += len(entity) - len(entity.lstrip())
        ent_end_char -= len(entity) - len(entity.rstrip())

        return text, ent_start_char, ent_end_char

    @staticmethod
    def _fill_ent_placeholder(
        self, text: str, ent_start_char: int, ent_end_char: int, placeholder: str
    ) -> Tuple[str, int, int]:
        text = text[0:ent_start_char] + placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(placeholder)

        return text, ent_start_char, ent_end_char

    def _get_negation_prob(
        self, text: str, ent_start_char: int, ent_end_char: int, probas_aggregator: Callable
    ) -> float:
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probas = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1) + 1

        return probas_aggregator(probas[start_token:end_token])

    def detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self._get_ent_window(ent, token_window=self.token_window)

            if self.strip_entities:
                text, ent_start_char, ent_end_char = self._trim_ent_boundaries(text, ent_start_char, ent_end_char)

            if self.placeholder is not None:
                text, ent_start_char, ent_end_char = self._fill_ent_placeholder(
                    text, ent_start_char, ent_end_char, placeholder=self.placeholder
                )

            if (
                self._get_negation_prob(text, ent_start_char, ent_end_char, probas_aggregator=self.probas_aggregator)
                > self.threshold
            ):
                self.add_qualifier_to_ent(ent, self.negation_qualifier.Negated)

        return doc
