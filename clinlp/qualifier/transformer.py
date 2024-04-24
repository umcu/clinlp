import statistics
from typing import Callable, Optional, Tuple

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

TRANSFORMER_NEGATION_REPO = "UMCU/MedRoBERTa.nl_NegationDetection"
TRANSFORMER_NEGATION_REVISION = "83068ba132b6ce38e9f668c1e3ab636f79b774d3"

TRANSFORMER_EXPERIENCER_REPO = "UMCU/MedRoBERTa.nl_Experiencer"
TRANSFORMER_EXPERIENCER_REVISION = "d9318c4b2b0ab0dfe50afedca58319b2369f1a71"

_defaults_negation_transformer = {
    "token_window": 32,
    "strip_entities": True,
    "placeholder": None,
    "probas_aggregator": statistics.mean,
    "negation_threshold": 0.5,
}
_defaults_experiencer_transformer = {
    "token_window": 64,
    "strip_entities": True,
    "placeholder": None,
    "probas_aggregator": statistics.mean,
    "patient_threshold": 0.5,
}


class QualifierTransformer(QualifierDetector):
    @staticmethod
    def _get_ent_window(ent: Span, token_window: int) -> Tuple[str, int, int]:
        start_token_i = max(0, ent.start - token_window)
        end_token_i = min(len(ent.doc), ent.end + token_window)

        text_span = ent.doc[start_token_i:end_token_i]

        ent_start_char = ent.start_char - text_span.start_char
        ent_end_char = ent.end_char - text_span.start_char

        return str(text_span), ent_start_char, ent_end_char

    @staticmethod
    def _trim_ent_boundaries(
        text: str, ent_start_char: int, ent_end_char: int
    ) -> Tuple[str, int, int]:
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


@Language.factory(
    name="clinlp_negation_transformer",
    requires=["doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_negation_transformer,
)
@clinlp_autocomponent
class NegationTransformer(QualifierTransformer):
    def __init__(
        self,
        nlp: Language,
        token_window: int = _defaults_negation_transformer["token_window"],
        strip_entities: bool = _defaults_negation_transformer["strip_entities"],
        placeholder: Optional[str] = _defaults_negation_transformer["placeholder"],
        probas_aggregator: Callable = _defaults_negation_transformer[
            "probas_aggregator"
        ],
        negation_threshold: float = _defaults_negation_transformer[
            "negation_threshold"
        ],
    ) -> None:
        self.nlp = nlp
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.probas_aggregator = probas_aggregator
        self.negation_threshold = negation_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_NEGATION_REPO, revision=TRANSFORMER_NEGATION_REVISION
        )
        self.model = RobertaForTokenClassification.from_pretrained(
            TRANSFORMER_NEGATION_REPO, revision=TRANSFORMER_NEGATION_REVISION
        )

    @property
    def qualifier_factories(self) -> dict[str, QualifierFactory]:
        return {
            "Negation": QualifierFactory(
                "Negation", ["Affirmed", "Unknown", "Negated"], default="Affirmed"
            )
        }

    def _get_negation_prob(
        self,
        text: str,
        ent_start_char: int,
        ent_end_char: int,
        probas_aggregator: Callable,
    ) -> float:
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probas = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return probas_aggregator(
            pos[0] + pos[2] for pos in probas[start_token : end_token + 1]
        )

    def _detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self._get_ent_window(
                ent, token_window=self.token_window
            )

            if self.strip_entities:
                text, ent_start_char, ent_end_char = self._trim_ent_boundaries(
                    text, ent_start_char, ent_end_char
                )

            if self.placeholder is not None:
                text, ent_start_char, ent_end_char = self._fill_ent_placeholder(
                    text, ent_start_char, ent_end_char, placeholder=self.placeholder
                )

            prob = self._get_negation_prob(
                text,
                ent_start_char,
                ent_end_char,
                probas_aggregator=self.probas_aggregator,
            )

            if prob > self.negation_threshold:
                self.add_qualifier_to_ent(
                    ent,
                    self.qualifier_factories["Negation"].create("Negated", prob=prob),
                )


@Language.factory(
    name="clinlp_experiencer_transformer",
    requires=["doc.ents"],
    assigns=[f"span._.{ATTR_QUALIFIERS}"],
    default_config=_defaults_experiencer_transformer,
)
@clinlp_autocomponent
class ExperiencerTransformer(QualifierTransformer):
    def __init__(
        self,
        nlp: Language,
        token_window: int = _defaults_experiencer_transformer["token_window"],
        strip_entities: bool = _defaults_experiencer_transformer["strip_entities"],
        placeholder: Optional[str] = _defaults_experiencer_transformer["placeholder"],
        probas_aggregator: Callable = _defaults_experiencer_transformer[
            "probas_aggregator"
        ],
        patient_threshold: float = _defaults_experiencer_transformer[
            "patient_threshold"
        ],
    ) -> None:
        self.nlp = nlp
        self.token_window = token_window
        self.strip_entities = strip_entities
        self.placeholder = placeholder
        self.probas_aggregator = probas_aggregator
        self.patient_threshold = patient_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_EXPERIENCER_REPO, revision=TRANSFORMER_EXPERIENCER_REVISION
        )
        self.model = RobertaForTokenClassification.from_pretrained(
            TRANSFORMER_EXPERIENCER_REPO, revision=TRANSFORMER_EXPERIENCER_REVISION
        )

    @property
    def qualifier_factories(self) -> dict[str, QualifierFactory]:
        return {
            "Experiencer": QualifierFactory(
                "Experiencer", ["Patient", "Unknown", "Other"], default="Patient"
            )
        }

    def _get_patient_prob(
        self,
        text: str,
        ent_start_char: int,
        ent_end_char: int,
        probas_aggregator: Callable,
    ) -> float:
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.forward(inputs["input_ids"])
        probas = torch.nn.functional.softmax(output.logits[0], dim=1).detach().numpy()

        start_token = inputs.char_to_token(ent_start_char)
        end_token = inputs.char_to_token(ent_end_char - 1)

        return probas_aggregator(
            pos[0] + pos[2] for pos in probas[start_token : end_token + 1]
        )

    def _detect_qualifiers(self, doc: Doc):
        for ent in doc.ents:
            text, ent_start_char, ent_end_char = self._get_ent_window(
                ent, token_window=self.token_window
            )

            if self.strip_entities:
                text, ent_start_char, ent_end_char = self._trim_ent_boundaries(
                    text, ent_start_char, ent_end_char
                )

            if self.placeholder is not None:
                text, ent_start_char, ent_end_char = self._fill_ent_placeholder(
                    text, ent_start_char, ent_end_char, placeholder=self.placeholder
                )

            prob = self._get_patient_prob(
                text,
                ent_start_char,
                ent_end_char,
                probas_aggregator=self.probas_aggregator,
            )

            if prob < self.patient_threshold:
                self.add_qualifier_to_ent(
                    ent,
                    self.qualifier_factories["Experiencer"].create(
                        "Other", prob=prob
                    ),
                )
