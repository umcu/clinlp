import statistics
from typing import Callable, Optional, Tuple, List

import torch
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, RobertaForTokenClassification

from clinlp.qualifier.qualifier import (
    ATTR_QUALIFIERS,
    QualifierDetector,
    QualifierFactory,
)
from clinlp.util import clinlp_autocomponent

TRANSFORMER_REPO = "UMCU/MedRoBERTa.nl_NegationDetection"

_defaults_negation_transformer = {
    "token_window": 32,
    "strip_entities": True,
    "placeholder": None,
    "probas_aggregator": statistics.mean,
    "negation_threshold": 0.5,
    "affirmed_threshold": 0.5,
}


@Language.factory(
    name="clinlp_negation_transformer",
    requires=["doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_negation_transformer,
)
@clinlp_autocomponent
class NegationTransformer(QualifierDetector):
    def __init__(
        self,
        nlp: Language,
        token_window: int = _defaults_negation_transformer["token_window"],
        strip_entities: bool = _defaults_negation_transformer["strip_entities"],
        placeholder: Optional[str] = _defaults_negation_transformer["placeholder"],
        probas_aggregator: Callable = _defaults_negation_transformer["probas_aggregator"],
        negation_threshold: float = _defaults_negation_transformer["negation_threshold"],
        affirmed_threshold: float = _defaults_negation_transformer["affirmed_threshold"],
    ):
        self.nlp = nlp
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.probas_aggregator = probas_aggregator
        self.negation_threshold = negation_threshold
        self.affirmed_threshold = affirmed_threshold

        self.negation_factory = QualifierFactory("Negation", ["AFFIRMED", "UNKNOWN", "NEGATED"])

        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_REPO)
        self.model = RobertaForTokenClassification.from_pretrained(TRANSFORMER_REPO)

    @staticmethod
    def _get_ent_window(ent: Span, token_window: int) -> Tuple[str, int, int]:
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
        text: str, ent_start_char: int, ent_end_char: int, placeholder: str
    ) -> Tuple[str, int, int]:
        text = text[0:ent_start_char] + placeholder + text[ent_end_char:]
        ent_end_char = ent_start_char + len(placeholder)

        return text, ent_start_char, ent_end_char

    # INPROGRESS: add batch processing
    def _get_negation_prob_batch(self, texts: List[str], ent_indices: List[Tuple[int, int]], probas_aggregator: Callable) -> List[float]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        output = self.model(**inputs)
        probas = torch.nn.functional.softmax(output.logits, dim=-1).detach().numpy()

        results = []
        for i, (start_char, end_char) in enumerate(ent_indices):
            start_token = inputs[i].char_to_token(start_char)
            end_token = inputs[i].char_to_token(end_char - 1)

            results.append(probas_aggregator(pos[0] + pos[2] for pos in probas[i][start_token : end_token + 1]))

        return results

    def _get_negation_prob(
        self, text: str, ent_start_char: int, ent_end_char: int, probas_aggregator: Callable
    ) -> float:
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probas = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return probas_aggregator(pos[0] + pos[2] for pos in probas[start_token : end_token + 1])

    #TODO: add batch processing to perform  self._get_ent_window,
    # self._trim_ent_boundaries and self._fill_ent_placeholder

    def detect_qualifiers(self, doc: Doc):
        # TODO: perhaps good to add a document-level qualifier that does not rely on the entities and instead
        #  aggregates the proba over strided spans with fixed length, and subsequently give a document-level qualification
        #  I think we then have to add .cats qualifiers instead of token/span qualifiers

        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self._get_ent_window(ent, token_window=self.token_window)

            if self.strip_entities:
                text, ent_start_char, ent_end_char = self._trim_ent_boundaries(text, ent_start_char, ent_end_char)

            if self.placeholder is not None:
                text, ent_start_char, ent_end_char = self._fill_ent_placeholder(
                    text, ent_start_char, ent_end_char, placeholder=self.placeholder
                )

            proba = self._get_negation_prob(text, ent_start_char, ent_end_char, probas_aggregator=self.probas_aggregator)
            if ( proba > self.negation_threshold ):
                self.add_qualifier_to_ent(ent, self.negation_qualifier.NEGATED, proba)
            elif ( 1-proba > self.affirmed_threshold ):
                self.add_qualifier_to_ent(ent, self.negation_qualifier.AFFIRMED, proba)
            else:
                self.add_qualifier_to_ent(ent, self.negation_qualifier.UNKNOWN, proba)
        return doc
